"""Compare fp32, fp16 and bf16 variants of one exported model.

For each precision this reports how many transcripts are identical to the fp32 ones,
the decode time and the process's peak GPU memory. Inputs are audio files (agreement
only) or a labelled pickle of [{"id", "ref", "wav"}] with mono 16 kHz float32 waveforms,
which adds WER and CER after Unicode normalization, lowercasing and punctuation removal.
LLM inputs given as `clip.wav:eng_Latn` are transcribed with that language code.

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
import time
import unicodedata
from pathlib import Path

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
    pairs = [(normalize(r), normalize(h)) for r, h in zip(references, hypotheses) if normalize(r)]
    words = sum(edits(r.split(), h.split()) for r, h in pairs)
    chars = sum(edits(r, h) for r, h in pairs)
    return {
        "wer": 100 * words / sum(len(r.split()) for r, _ in pairs),
        "cer": 100 * chars / sum(len(r) for r, _ in pairs),
        "n": len(pairs),
    }


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


def load_rows(args):
    if args.dataset:
        return pickle.loads(args.dataset.read_bytes())[: args.limit]
    from fast_omniasr.audio import load_audio

    rows = []
    for spec in args.audio:
        path, _, lang = spec.partition(":")
        rows.append({"id": spec, "wav": load_audio(path)[0], "lang": lang or None})
    return rows


def transcriber(args, precision):
    if args.llm:
        from fast_omniasr.llm import OmniASRLLM

        model = OmniASRLLM(args.llm, backend="onnx", device=args.device, precision=precision)
        return lambda row: model.transcribe(row["wav"], lang=row.get("lang") or args.lang).text
    from fast_omniasr import OmniASR

    graph = args.ctc / ("model.onnx" if precision == "fp32" else f"{precision}/model.onnx")
    model = OmniASR(graph, args.ctc / "tokenizer.model", device=args.device, precision=precision)
    return lambda row: model.transcribe_numpy(row["wav"]).text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--llm", type=Path, help="LLM export directory")
    source.add_argument("--ctc", type=Path, help="CTC export directory (model.onnx, tokenizer.model)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", nargs="+", help="Audio files, optionally `path:lang_code`")
    inputs.add_argument("--dataset", type=Path, help="Labelled pickle (adds WER/CER)")
    parser.add_argument("--lang", help="Default LLM language code (default: none)")
    parser.add_argument("--precision", nargs="+", default=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.precision[0] != "fp32":
        raise SystemExit("Evaluate fp32 first; agreement is measured against it")
    rows = load_rows(args)
    results, baseline = {}, None
    for precision in args.precision:
        run = transcriber(args, precision)
        run(rows[0])  # warm-up: weights load and CUDA kernels initialize
        start = time.perf_counter()
        hypotheses = [run(row) for row in rows]
        seconds = time.perf_counter() - start
        baseline = baseline or hypotheses
        results[precision] = {
            **(error_rates([row["ref"] for row in rows], hypotheses) if args.dataset else {}),
            "inputs": len(rows),
            "identical_to_fp32": sum(a == b for a, b in zip(baseline, hypotheses)),
            "seconds": seconds,
            "gpu_mib": gpu_mib(),
            "hypotheses": dict(zip((row["id"] for row in rows), hypotheses)),
        }
        summary = {k: v for k, v in results[precision].items() if k != "hypotheses"}
        print(json.dumps({"precision": precision, **summary}), flush=True)
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
