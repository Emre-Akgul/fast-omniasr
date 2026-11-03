"""Compare fp32, fp16 and bf16 variants of one exported model.

For each precision this reports how many transcripts are identical to the fp32 ones,
the decode time and the peak GPU memory its process used while loading, warming up and
transcribing (sampled from nvidia-smi throughout; null when it can't be measured). By default each
precision runs in a fresh process (--no-isolate shares one), so one precision's memory
arena can't inflate the next one's peak. Inputs are audio files (agreement
only) or a labelled dataset, which adds WER and CER after Unicode normalization,
lowercasing and punctuation removal. A dataset is a JSONL manifest, one object per line:

  {"id": "en-001", "audio": "clips/en-001.flac", "ref": "Hello there.", "lang": "eng_Latn"}

`audio` is a sound file or a `.npy` array (loaded without pickle; give "sample_rate" if it
isn't 16000), relative to the manifest's directory; `ref` may be null for silence and
`lang` is optional. Pickled datasets ([{"id", "ref", "wav"}]) can run arbitrary code when
loaded, so they are only read with --trust-pickle.
Utterances whose normalized reference is empty (silence, noise) are kept: any words
emitted for them count as insertions, and they are also summarized separately.
LLM inputs given as `clip.wav:eng_Latn` are transcribed with that language code.
For LLM models each precision also reports why generation stopped: `eos` (complete),
`max_new_tokens` or `context_limit` (truncated). Truncated transcripts are listed by id
and flagged on stderr, since they are not comparable with complete ones.

  python tools/evaluate_precisions.py --llm /path/to/llm-export \\
      --audio en.wav en.wav:eng_Latn tr.wav:tur_Latn --report precision.json
  python tools/evaluate_precisions.py --ctc /path/to/ctc-export --audio en.wav tr.wav
  python tools/evaluate_precisions.py --ctc /path/to/ctc-export --dataset eval.jsonl
"""

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
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


def gpu_mib(pid=None):
    """GPU memory held by a process (MiB) per nvidia-smi; None if it can't be measured:
    nvidia-smi is missing or fails, reports no usable number (e.g. "[N/A]"), or doesn't list
    the process (CPU-only runs, containers with their own PID namespace)."""
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    usage = None
    for line in out.strip().splitlines():
        try:
            row_pid, mib = (int(field) for field in line.split(","))
        except ValueError:
            continue  # "[N/A]" or another unparsable row
        if row_pid == pid:
            usage = (usage or 0) + mib
    return usage


def _default_query():
    return gpu_mib()  # looked up at call time, so tests can patch gpu_mib


class GpuMemorySampler:
    """Track this process's peak GPU memory while the block runs, by polling `query`
    (default: nvidia-smi) from a background thread. `peak` is None if unavailable."""

    def __init__(self, interval=0.25, query=None):
        self.interval = interval
        self.query = query if query is not None else _default_query
        self.peak = None
        self._stop = threading.Event()

    def _sample(self):
        value = self.query()
        if value is not None:
            self.peak = value if self.peak is None else max(self.peak, value)
        return value

    def _poll(self):
        while not self._stop.wait(self.interval):
            self._sample()  # a failed reading (None) is skipped, not fatal

    def __enter__(self):
        # Before model load the process may not be listed yet, so poll unconditionally,
        # unless nvidia-smi itself is missing.
        self._sample()
        if shutil.which("nvidia-smi") or self.query is not _default_query:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join()
        self._sample()  # final reading even when no polling thread ran


def run_precision(args, precision, rows):
    """Load, warm up and transcribe every row at one precision, measuring peak GPU memory
    over the whole run (weights and kernels included)."""
    with GpuMemorySampler() as memory:
        run = transcriber(args, precision)
        run(rows[0])  # warm-up: weights load and CUDA kernels initialize
        start = time.perf_counter()
        outputs = [run(row) for row in rows]
        seconds = time.perf_counter() - start
        del run
    return {"texts": [text for text, _ in outputs], "reasons": [r for _, r in outputs],
            "seconds": seconds, "peak_gpu_mib": memory.peak}


def run_in_fresh_process(function, *args):
    """Call `function(*args)` in a newly spawned interpreter and return its result, so GPU
    memory (and arenas) from one call can't affect another."""
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(function, args)


LLM_MAX_SECONDS = 30  # OmniASRLLM.transcribe rejects longer clips


def validate_rows(rows, labelled, max_seconds=None):
    """Fail before any model loads: a non-empty sequence of rows with unique, hashable ids,
    a reference for every labelled row (a string, or None for silence), and 16 kHz mono
    audio that is long enough, finite and, with `max_seconds`, short enough."""
    from collections.abc import Hashable

    from fast_omniasr.audio import prepare_audio

    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)):
        raise TypeError("evaluation inputs must be a list of rows")
    rows = list(rows)
    if not rows:
        raise ValueError("no evaluation inputs: the dataset or --audio list is empty")
    seen, duplicates = set(), []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "id" not in row or "wav" not in row:
            raise TypeError(f"row {index} must be a dict with 'id' and 'wav'")
        if not isinstance(row["id"], Hashable):
            raise TypeError(f"row {index} has an unhashable id {row['id']!r}")
        if row["id"] in seen:
            duplicates.append(row["id"])
        seen.add(row["id"])
        if labelled and not (row.get("ref") is None and "ref" in row
                             or isinstance(row.get("ref"), str)):
            raise ValueError(f"row {row['id']!r} has no 'ref' (reference transcript; "
                             "None for silence)")
        if row.get("lang") is not None and not isinstance(row["lang"], str):
            raise ValueError(f"row {row['id']!r} has a non-string 'lang' {row['lang']!r}")
        try:
            audio = prepare_audio(row["wav"], row.get("sample_rate", 16000))
        except ValueError as exc:
            raise ValueError(f"row {row['id']!r}: {exc}") from exc
        if max_seconds is not None and audio.shape[1] > max_seconds * 16000:
            raise ValueError(f"row {row['id']!r} is longer than the LLM runtime's "
                             f"{max_seconds} s limit")
    if duplicates:
        raise ValueError(f"duplicate ids would overwrite each other's results: {duplicates}")
    return rows


_MANIFEST_KEYS = {"id", "audio", "ref", "lang", "sample_rate"}


def read_manifest(path, limit=None):
    """Rows from a JSONL manifest; audio is read from files (sound files, or .npy arrays
    loaded with allow_pickle=False) resolved against the manifest's directory."""
    import numpy as np
    import soundfile as sf

    rows = []
    with Path(path).open(encoding="utf-8-sig") as f:  # tolerate a BOM from Windows editors
        lines = enumerate(f, 1)
        while True:
            try:
                number, line = next(lines)
            except StopIteration:
                break
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{path} is not a UTF-8 JSONL manifest (if it is a pickled dataset, "
                    "convert it, or name it .pkl and pass --trust-pickle)"
                ) from exc
            if limit is not None and len(rows) == limit:
                break
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(entry, dict) or not {"id", "audio"} <= entry.keys():
                raise ValueError(f"{path}:{number}: each line needs at least 'id' and 'audio'")
            unknown = entry.keys() - _MANIFEST_KEYS
            if unknown:
                raise ValueError(f"{path}:{number}: unknown keys {sorted(unknown)}")
            if not isinstance(entry["audio"], str):
                raise TypeError(f"{path}:{number}: 'audio' must be a file path string")
            audio = Path(path).parent / entry["audio"]
            is_array = audio.suffix.lower() == ".npy"
            try:
                if is_array:
                    waveform = np.load(audio, allow_pickle=False)
                    sample_rate = entry.get("sample_rate", 16000)
                else:
                    waveform, sample_rate = sf.read(audio, dtype="float32")
            except ValueError:
                raise  # e.g. a pickled .npy refused by allow_pickle=False
            except Exception as exc:  # missing or unreadable files, in any audio format
                raise ValueError(f"{path}:{number}: cannot read {audio}: {exc}") from exc
            if not is_array and entry.get("sample_rate") not in (None, sample_rate):
                raise ValueError(f"{path}:{number}: sample_rate {entry['sample_rate']} "
                                 f"disagrees with {audio.name} ({sample_rate} Hz)")
            row = {"id": entry["id"], "wav": waveform, "sample_rate": sample_rate,
                   "lang": entry.get("lang")}
            if "ref" in entry:
                row["ref"] = entry["ref"]
            rows.append(row)
    return rows


# eng_Latn, plus dialect-qualified codes such as roh_Latn_surs1244.
_LANG_CODE = re.compile(r"[a-z]{3}_[A-Za-z]{4}(?:_[A-Za-z0-9]+)?")


def _split_audio_spec(spec):
    """'clip.wav:eng_Latn' -> ('clip.wav', 'eng_Latn'); a colon that isn't followed by a
    language code (e.g. a Windows drive, 'C:\\clips\\a.wav') is part of the path."""
    path, sep, lang = spec.rpartition(":")
    if sep and (lang == "" or _LANG_CODE.fullmatch(lang)) and path:
        return path, lang  # "clip.wav:" means no language, as before
    return spec, ""


def load_rows(args):
    max_seconds = LLM_MAX_SECONDS if getattr(args, "llm", None) else None
    if args.dataset:
        if args.dataset.suffix.lower() in (".pkl", ".pickle"):
            if not getattr(args, "trust_pickle", False):
                raise ValueError(
                    f"{args.dataset} is a pickle, which can run arbitrary code when loaded; "
                    "convert it to a JSONL manifest, or pass --trust-pickle if you trust it"
                )
            rows = pickle.loads(args.dataset.read_bytes())
            if not isinstance(rows, (list, tuple)):
                raise TypeError(f"{args.dataset} must contain a list of rows")
            rows = rows[: args.limit]
        else:
            rows = read_manifest(args.dataset, args.limit)
        return validate_rows(rows, labelled=True, max_seconds=max_seconds)
    import soundfile as sf

    rows = []
    for spec in args.audio[: args.limit]:  # --limit before reading any file
        path, lang = _split_audio_spec(spec)
        # Raw samples: the transcribers normalize exactly once, like OmniASR.transcribe(path).
        waveform, sample_rate = sf.read(path, dtype="float32")  # validated in validate_rows
        rows.append({"id": spec, "wav": waveform, "sample_rate": sample_rate,
                     "lang": lang or None})
    return validate_rows(rows, labelled=False, max_seconds=max_seconds)


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
    inputs.add_argument("--dataset", type=Path,
                        help="Labelled JSONL manifest (adds WER/CER); see the module docstring")
    parser.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=True,
                        help="run each precision in a fresh process (default) so peak GPU "
                             "memory is measured per precision")
    parser.add_argument("--trust-pickle", action="store_true",
                        help="allow a .pkl --dataset (unpickling can run arbitrary code)")
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
    try:
        rows = load_rows(args)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    results, baseline = {}, None
    for precision in args.precision:
        if args.isolate:
            measured = run_in_fresh_process(run_precision, args, precision, rows)
        else:
            measured = run_precision(args, precision, rows)
        hypotheses, reasons, seconds = measured["texts"], measured["reasons"], measured["seconds"]
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
            "peak_gpu_mib": measured["peak_gpu_mib"],
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
    report = {
        "model": str(args.llm or args.ctc),
        "inputs": args.dataset.name if args.dataset else args.audio,
        "lang": args.lang,
        "device": args.device,
        "normalization": "NFKC, lowercase, zero-width and punctuation removed",
        "precisions": results,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
