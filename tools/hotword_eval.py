"""Measure what hotwords do on your own recordings, false insertions included.

    python tools/hotword_eval.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX \\
        --manifest clips.jsonl --hotwords names.txt --lang tur_Latn --max-suffix 5

Each manifest line is {"audio": path, "text": transcript}; the hotwords file has one per line.
Include clips that contain none of the hotwords: they are where false insertions show.

For greedy decoding, beam search without hotwords and each --weights value it reports:
- wer: case- and punctuation-insensitive word error rate (percent);
- found: hotword occurrences of the transcripts that were recognized, out of all of them;
- inserted: hotword occurrences recognized where the transcript has none (in excess of the
  transcript's count, per clip and hotword);
- ms: decoding time per clip (the model itself excluded)."""

import argparse
import json
import time
import unicodedata
from pathlib import Path

from fast_omniasr import BeamSearchDecoder
from fast_omniasr.beam_search import _fold_text, _is_boundary
from fast_omniasr.decoder import greedy_token_ids


def words(text: str, lang: str | None) -> list[str]:
    folded = _fold_text(unicodedata.normalize("NFC", text), lang)
    return "".join(" " if _is_boundary(c) else c for c in folded if c != "'").split()


def edits(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row = row, [i]
        for j, y in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (x != y)))
    return row[-1]


def score(references, hypotheses, counter: BeamSearchDecoder) -> dict:
    """WER, and the hotwords of `counter` found and wrongly inserted."""
    errors = total = found = said = inserted = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        expected = words(reference, counter.lang)
        errors += edits(expected, words(hypothesis, counter.lang))
        total += len(expected)
        want, got = counter.hotword_counts(reference), counter.hotword_counts(hypothesis)
        for hotword, count in want.items():
            said += count
            found += min(count, got[hotword])
            inserted += max(0, got[hotword] - count)
    return {"wer": round(100 * errors / max(1, total), 2), "found": found, "said": said,
            "inserted": inserted}


def evaluate(logits, references, tokenizer, hotwords, *, weights=(2.0, 5.0, 10.0),
             **options) -> list[dict]:
    """One row per decoding; `options` go to BeamSearchDecoder."""
    counter = BeamSearchDecoder(hotwords=hotwords, lang=options.get("lang"),
                                hotword_max_suffix=options.get("hotword_max_suffix", 0))
    runs = [("greedy", None), ("beam", BeamSearchDecoder(**options))]
    runs += [(f"hotwords {weight:g}",
              BeamSearchDecoder(hotwords=hotwords, hotword_weight=weight, **options))
             for weight in weights]
    rows = []
    for name, decoder in runs:
        started = time.perf_counter()
        ids = [greedy_token_ids(clip) if decoder is None else decoder.token_ids(clip, tokenizer)
               for clip in logits]
        elapsed = time.perf_counter() - started
        texts = [tokenizer.decode(clip) for clip in ids]
        rows.append({"decoding": name, **score(references, texts, counter),
                     "ms": round(1000 * elapsed / max(1, len(logits)), 1), "texts": texts})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hotwords", type=Path, required=True)
    parser.add_argument("--lang")
    parser.add_argument("--max-suffix", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--weights", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    args = parser.parse_args()

    from fast_omniasr import OmniASR
    from fast_omniasr.audio import load_audio

    clips = [json.loads(line) for line in args.manifest.read_text("utf-8").splitlines() if line]
    hotwords = [line for line in args.hotwords.read_text("utf-8").splitlines() if line.strip()]
    with OmniASR.from_pretrained(args.model) as model:
        logits = [model.backend.infer(load_audio(clip["audio"])) for clip in clips]
        rows = evaluate(logits, [clip["text"] for clip in clips], model.tokenizer, hotwords,
                        weights=args.weights, beam_size=args.beam_size, lang=args.lang,
                        hotword_max_suffix=args.max_suffix)
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != "texts"}))


if __name__ == "__main__":
    main()
