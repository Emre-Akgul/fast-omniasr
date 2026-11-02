"""Compare fp32, fp16 and bf16 variants of one exported model.

For each precision this reports how many transcripts are identical to the fp32 ones,
the decode time and the process's peak GPU memory. Inputs are audio files (agreement
only) or a labelled pickle of [{"id", "ref", "wav"}] with mono 16 kHz float32 waveforms,
which adds WER and CER after Unicode normalization, lowercasing and punctuation removal.
Utterances whose normalized reference is empty (silence, noise) are kept: any words
emitted for them count as insertions, and they are also summarized separately.
LLM inputs given as `clip.wav:eng_Latn` are transcribed with that language code.
For LLM models each precision also reports why generation stopped: `eos` (complete),
`max_new_tokens` or `context_limit` (truncated). Truncated transcripts are listed by id
and flagged on stderr, since they are not comparable with complete ones.

  python tools/evaluate_precisions.py --llm /path/to/llm-export \\
      --audio en.wav en.wav:eng_Latn tr.wav:tur_Latn --report precision.json
  python tools/evaluate_precisions.py --ctc /path/to/ctc-export --audio en.wav tr.wav
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from fast_omniasr.llm.runtime import STOP_REASONS as LLM_STOP_REASONS

_PUNCT = re.compile(r"[^\w\s]|_")


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("\u0307", "")  # combining dot left by lowercasing "\u0130"
    text = re.sub("[\u200b-\u200f\u2060\ufeff\u00ad]", "", text)  # zero-width, soft hyphen
    text = text.replace("\u2019", "'").replace("'", "")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def edits(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def error_rates(references, hypotheses):
    """Corpus WER/CER over every utterance, sclite/jiwer-style: errors summed over all
    pairs divided by the total reference length. Empty references (after normalization)
    are not dropped: output on silence adds insertions to the numerator and nothing to
    the denominator, so hallucinations stay visible. If every reference is empty the
    rates are undefined and reported as None, with the insertions still counted."""
    references, hypotheses = list(references), list(hypotheses)
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    pairs = [(normalize(r), normalize(h)) for r, h in zip(references, hypotheses)]
    ref_words = sum(len(r.split()) for r, _ in pairs)
    ref_chars = sum(len(r) for r, _ in pairs)
    words = sum(edits(r.split(), h.split()) for r, h in pairs)
    chars = sum(edits(r, h) for r, h in pairs)
    silent = [h for r, h in pairs if not r]
    return {
        "wer": 100 * words / ref_words if ref_words else None,
        "cer": 100 * chars / ref_chars if ref_chars else None,
        "n": len(pairs),
        "empty_references": len(silent),
        # Utterances with an empty reference that still produced (normalized) text.
        "empty_reference_nonempty_hypotheses": sum(bool(h) for h in silent),
        "empty_reference_inserted_words": sum(len(h.split()) for h in silent),
    }


def stop_reason_summary(ids, reasons):
    """Count LLM stop reasons and list the ids of transcripts that did not end at EOS."""
    unknown = sorted({r for r in reasons if r not in LLM_STOP_REASONS}, key=str)
    if unknown:
        raise ValueError(f"unexpected stop reasons: {unknown}")
    counts = {reason: 0 for reason in LLM_STOP_REASONS}
    for reason in reasons:
        counts[reason] += 1
    truncated = [i for i, r in zip(ids, reasons) if r != "eos"]
    return {"stop_reasons": counts, "truncated": len(truncated), "truncated_ids": truncated}


def gpu_mib():
    """GPU memory held by this process (MiB), from nvidia-smi; 0 if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    rows = (line.split(",") for line in out.strip().splitlines())
    return sum(int(mib) for pid, mib in rows if int(pid) == os.getpid())


def validate_rows(rows, labelled):
    """Fail before any model loads: non-empty, unique ids, a reference for every labelled
    row, and audio that every runtime accepts (16 kHz mono, long enough, finite)."""
    from fast_omniasr.audio import prepare_audio

    if not isinstance(rows, list) or not rows:
        raise ValueError("no evaluation inputs: the dataset or --audio list is empty")
    seen, duplicates = set(), []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "id" not in row or "wav" not in row:
            raise ValueError(f"row {index} must be a dict with 'id' and 'wav'")
        if row["id"] in seen:
            duplicates.append(row["id"])
        seen.add(row["id"])
        if labelled and not isinstance(row.get("ref"), str):
            raise ValueError(f"row {row['id']!r} has no string 'ref' (reference transcript)")
        try:
            prepare_audio(row["wav"], row.get("sample_rate", 16000))
        except ValueError as exc:
            raise ValueError(f"row {row['id']!r}: {exc}") from exc
    if duplicates:
        raise ValueError(f"duplicate ids would overwrite each other's results: {duplicates}")
    return rows


def load_rows(args):
    if args.dataset:
        rows = pickle.loads(args.dataset.read_bytes())
        if not isinstance(rows, list):
            raise ValueError(f"{args.dataset} must contain a list of rows")
        return validate_rows(rows[: args.limit], labelled=True)
    import soundfile as sf

    rows = []
    for spec in args.audio:
        path, _, lang = spec.partition(":")
        # Raw samples: the transcribers normalize exactly once, like OmniASR.transcribe(path).
        waveform, sample_rate = sf.read(path, dtype="float32")  # validated in validate_rows
        rows.append({"id": spec, "wav": waveform, "sample_rate": sample_rate,
                     "lang": lang or None})
    return validate_rows(rows[: args.limit], labelled=False)


def transcriber(args, precision):
    """Return row -> (text, stop_reason). CTC decoding has no stop reason (None)."""
    if args.llm:
        from fast_omniasr.llm import OmniASRLLM

        model = OmniASRLLM(args.llm, backend="onnx", device=args.device, precision=precision)

        def run_llm(row):
            result = model.transcribe(row["wav"], sample_rate=row.get("sample_rate", 16000),
                                      lang=row.get("lang") or args.lang)
            return result.text, result.stop_reason

        return run_llm
    from fast_omniasr import OmniASR

    graph = args.ctc / ("model.onnx" if precision == "fp32" else f"{precision}/model.onnx")
    model = OmniASR(graph, args.ctc / "tokenizer.model", device=args.device, precision=precision)
    return lambda row: (
        model.transcribe_numpy(row["wav"], row.get("sample_rate", 16000)).text, None
    )


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--llm", type=Path, help="LLM export directory")
    source.add_argument("--ctc", type=Path, help="CTC export directory (model.onnx, tokenizer.model)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", nargs="+", help="Audio files, optionally `path:lang_code`")
    inputs.add_argument("--dataset", type=Path, help="Labelled pickle (adds WER/CER)")
    parser.add_argument("--lang", help="Default LLM language code (default: none)")
    parser.add_argument("--precision", nargs="+", choices=["fp32", "fp16", "bf16"],
                        default=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=_positive_int, help="first N inputs")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.precision[0] != "fp32":
        raise SystemExit("Evaluate fp32 first; agreement is measured against it")
    if len(set(args.precision)) != len(args.precision):
        raise SystemExit("--precision lists a precision twice; its results would be overwritten")
    rows = load_rows(args)
    results, baseline = {}, None
    for precision in args.precision:
        run = transcriber(args, precision)
        run(rows[0])  # warm-up: weights load and CUDA kernels initialize
        start = time.perf_counter()
        outputs = [run(row) for row in rows]
        seconds = time.perf_counter() - start
        hypotheses = [text for text, _ in outputs]
        reasons = [reason for _, reason in outputs]
        ids = [row["id"] for row in rows]
        baseline = baseline or hypotheses
        stops = stop_reason_summary(ids, reasons) if args.llm else {}
        results[precision] = {
            **(error_rates([row["ref"] for row in rows], hypotheses) if args.dataset else {}),
            **stops,
            "inputs": len(rows),
            # Word-level: CTC can emit whitespace-only output that differs in length only.
            "identical_to_fp32": sum(a.split() == b.split() for a, b in zip(baseline, hypotheses)),
            "seconds": seconds,
            "gpu_mib": gpu_mib(),
            "hypotheses": dict(zip(ids, hypotheses)),
            **({"hypothesis_stop_reasons": dict(zip(ids, reasons))} if args.llm else {}),
        }
        per_utterance = ("hypotheses", "hypothesis_stop_reasons", "truncated_ids")
        summary = {k: v for k, v in results[precision].items() if k not in per_utterance}
        print(json.dumps({"precision": precision, **summary}), flush=True)
        if stops.get("truncated"):
            print(f"warning: {precision}: {stops['truncated']}/{len(rows)} transcripts did not "
                  f"end at EOS ({stops['stop_reasons']}); ids: {stops['truncated_ids']}",
                  file=sys.stderr, flush=True)
        del run
    report = {
        "model": str(args.llm or args.ctc),
        "inputs": args.dataset.name if args.dataset else args.audio,
        "lang": args.lang,
        "device": args.device,
        "normalization": "NFKC, lowercase, zero-width and punctuation removed",
        "precisions": results,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
