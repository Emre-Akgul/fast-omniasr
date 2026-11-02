"""Per-language FLEURS WER/CER for the published models, with regression gates.

The evaluation set is pinned in benchmarks/fleurs/manifest.json: a FLEURS commit and, per
language, a fixed sample of test utterances with their reference transcripts and audio
sha256. Thresholds per model and language live in benchmarks/fleurs/thresholds.json.

  # score a model and fail if any language regresses past its threshold
  python tools/fleurs_eval.py run --model ctc --gate --report fleurs-ctc.json
  python tools/fleurs_eval.py run --model llm --languages en_us tr_tr --per-language 2

  # maintainers: rebuild the manifest from downloaded FLEURS TSVs
  python tools/fleurs_eval.py build-manifest --per-language 20

Only the listed utterances are extracted from each pinned language tarball; they are
verified against the manifest and kept under --audio-dir, so later runs skip the
download. Metrics are corpus WER/CER after the normalization in evaluate_precisions.py;
for languages gated on CER (Mandarin) whitespace is removed first, because normalization
turns CJK punctuation into spaces that would otherwise count as characters.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "fleurs" / "manifest.json"
THRESHOLDS = ROOT / "benchmarks" / "fleurs" / "thresholds.json"
DATASET = "google/fleurs"
MAX_SECONDS = 20  # well inside the LLM runtime's 30 s clip limit
_FILE_NAME = re.compile(r"[0-9]+\.wav")

# FLEURS language -> (OmniASR language code, gated metric). Mandarin isn't space-delimited,
# so its WER is meaningless and only CER is gated.
LANGUAGES = {
    "en_us": ("eng_Latn", "wer"),
    "tr_tr": ("tur_Latn", "wer"),
    "ckb_iq": ("ckb_Arab", "wer"),
    "de_de": ("deu_Latn", "wer"),
    "es_419": ("spa_Latn", "wer"),
    "fr_fr": ("fra_Latn", "wer"),
    "hi_in": ("hin_Deva", "wer"),
    "cmn_hans_cn": ("cmn_Hans", "cer"),
}

_spec = importlib.util.spec_from_file_location("evaluate_precisions",
                                               Path(__file__).with_name("evaluate_precisions.py"))
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
error_rates = _metrics.error_rates


# --- manifest -------------------------------------------------------------------------

def read_tsv(path: Path) -> list[dict]:
    """FLEURS TSV rows: id, file_name, raw_transcription, transcription, phonemes,
    num_samples, gender (no header)."""
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for fields in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(fields) != 7:
                raise ValueError(f"{path}: expected 7 tab-separated fields, got {len(fields)}")
            rows.append({"id": fields[0], "file_name": fields[1], "reference": fields[2],
                         "num_samples": int(fields[5])})
    return rows


def select_utterances(rows: list[dict], count: int, max_samples: int) -> list[dict]:
    """A deterministic sample: at most one recording per sentence id, short enough for
    every runtime, ordered by a hash of the file name so it doesn't follow speaker or
    sentence order."""
    usable = [r for r in rows if r["num_samples"] <= max_samples and r["reference"].strip()]
    usable.sort(key=lambda r: hashlib.sha256(r["file_name"].encode()).hexdigest())
    chosen, sentences = [], set()
    for row in usable:
        if row["id"] in sentences:
            continue
        sentences.add(row["id"])
        chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError(f"only {len(chosen)} usable utterances, wanted {count}")
    return chosen


def build_manifest(fleurs_root: Path, revision: str, per_language: int) -> dict:
    languages = {}
    for fleurs, (lang, metric) in LANGUAGES.items():
        rows = select_utterances(read_tsv(fleurs_root / fleurs / "test.tsv"), per_language,
                                 MAX_SECONDS * 16000)
        members = {f"test/{r['file_name']}" for r in rows}
        digests = {}
        with tarfile.open(fleurs_root / fleurs / "audio" / "test.tar.gz") as tar:
            for member in tar:
                if member.name in members:
                    digests[member.name] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
        languages[fleurs] = {
            "lang": lang, "metric": metric,
            "utterances": [{"file_name": r["file_name"], "reference": r["reference"],
                            "seconds": round(r["num_samples"] / 16000, 2),
                            "sha256": digests[f"test/{r['file_name']}"]} for r in rows],
        }
    return {"dataset": DATASET, "revision": revision, "split": "test", "license": "CC-BY-4.0",
            "languages": languages}


def validate_manifest(manifest: dict) -> None:
    if manifest.get("dataset") != DATASET or len(manifest.get("revision", "")) != 40:
        raise ValueError("manifest must pin google/fleurs to a 40-character commit")
    for fleurs, entry in manifest["languages"].items():
        if fleurs not in LANGUAGES:  # keys become paths and Hub filenames
            raise ValueError(f"unknown FLEURS language {fleurs!r}")
        if entry.get("metric") not in ("wer", "cer"):
            raise ValueError(f"{fleurs}: metric must be 'wer' or 'cer'")
        names = [u["file_name"] for u in entry["utterances"]]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"{fleurs}: utterances must be non-empty and unique")
        for utterance in entry["utterances"]:
            if not _FILE_NAME.fullmatch(utterance["file_name"]):  # it becomes a local path
                raise ValueError(f"{fleurs}: unexpected file name {utterance['file_name']!r}")
            if len(utterance.get("sha256", "")) != 64 or not utterance.get("reference"):
                raise ValueError(f"{fleurs}/{utterance.get('file_name')}: incomplete entry")


# --- audio ----------------------------------------------------------------------------

def fetch_audio(manifest: dict, audio_dir: Path, languages: list[str],
                per_language: int | None, cache_dir: Path | None = None) -> dict[str, list[Path]]:
    """Paths of the selected utterances, extracting any missing ones from the pinned
    tarball. Every file is checked against its manifest sha256."""
    paths = {}
    for fleurs in languages:
        utterances = manifest["languages"][fleurs]["utterances"][:per_language]
        target = audio_dir / fleurs
        missing = [u for u in utterances if not _verified(target / u["file_name"], u["sha256"])]
        if missing:
            from huggingface_hub import hf_hub_download

            archive = hf_hub_download(DATASET, f"data/{fleurs}/audio/test.tar.gz",
                                      repo_type="dataset", revision=manifest["revision"],
                                      cache_dir=cache_dir)
            extract_members(Path(archive), target, missing)
        paths[fleurs] = [target / u["file_name"] for u in utterances]
    return paths


def _verified(path: Path, sha256: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256


def extract_members(archive: Path, target: Path, utterances: list[dict]) -> None:
    wanted = {f"test/{u['file_name']}": u for u in utterances}
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            utterance = wanted.pop(member.name, None)
            if utterance is None:
                continue
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != utterance["sha256"]:
                raise RuntimeError(f"{member.name} in {archive.name} does not match the manifest")
            tmp = target / (utterance["file_name"] + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(target / utterance["file_name"])
    if wanted:
        raise RuntimeError(f"{archive.name} is missing {sorted(wanted)}")


# --- evaluation -----------------------------------------------------------------------

def score(references: list[str], hypotheses: list[str], metric: str = "wer") -> dict:
    """Corpus WER/CER. For CER-gated (unsegmented) languages, whitespace is dropped after
    normalization, so punctuation-turned-spaces and spacing choices don't count as
    errors, and WER is not reported."""
    if metric == "cer":
        def squash(texts):
            return [_metrics.normalize(t).replace(" ", "") for t in texts]

        rates = error_rates(squash(references), squash(hypotheses))
        return {"wer": None, "cer": rates["cer"], "n": rates["n"]}
    rates = error_rates(references, hypotheses)
    return {"wer": rates["wer"], "cer": rates["cer"], "n": rates["n"]}


def skipped_languages(results: dict, thresholds: dict) -> list[str]:
    """Thresholded languages that this run did not measure (e.g. a --languages subset)."""
    return sorted(set(thresholds.get("languages", {})) - set(results))


def check(results: dict, thresholds: dict, manifest: dict) -> list[str]:
    """Human-readable failures: languages whose gated metric (WER, plus CER always)
    exceeds its maximum. Languages without a threshold are reported, not gated."""
    failures = []
    for fleurs, measured in results.items():
        limits = thresholds.get("languages", {}).get(fleurs)
        if limits is None:
            continue
        gated = ["cer"] if manifest["languages"][fleurs]["metric"] == "cer" else ["wer", "cer"]
        for metric in gated:
            limit = limits.get(f"max_{metric}")
            value = measured.get(metric)
            if limit is not None and value is not None and value > limit:
                failures.append(f"{fleurs}: {metric.upper()} {value:.2f} > {limit:.2f}")
    return failures


def transcriber(model: str, repo: str, revision: str, cache_dir: Path | None):
    import soundfile as sf

    if model == "ctc":
        from fast_omniasr import OmniASR

        asr = OmniASR.from_pretrained(repo, revision=revision, device="cpu", cache_dir=cache_dir)

        def run(path, lang):
            waveform, rate = sf.read(path, dtype="float32")
            return asr.transcribe_numpy(waveform, rate).text, None
    else:
        from fast_omniasr.llm import OmniASRLLM

        asr = OmniASRLLM.from_pretrained(repo, revision=revision, device="cpu",
                                         cache_dir=cache_dir)

        def run(path, lang):
            waveform, rate = sf.read(path, dtype="float32")
            result = asr.transcribe(waveform, sample_rate=rate, lang=lang)
            return result.text, result.stop_reason
    return asr, run


def evaluate(args) -> int:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    thresholds = json.loads(args.thresholds.read_text())
    model = thresholds["models"][args.model]
    repo, revision = args.repo or model["repo"], args.revision or model["revision"]
    languages = args.languages or list(manifest["languages"])
    unknown = sorted(set(languages) - set(manifest["languages"]))
    if unknown:
        raise SystemExit(f"unknown --languages {unknown}; choose from {list(manifest['languages'])}")
    per_language = args.per_language or model.get("per_language")
    if args.gate and per_language != model.get("per_language"):
        raise SystemExit(f"--gate compares against baselines measured on {model['per_language']} "
                         f"utterances per language; don't pass --per-language {per_language}")
    paths = fetch_audio(manifest, args.audio_dir, languages, per_language, args.cache_dir)
    asr, run = transcriber(args.model, repo, revision, args.cache_dir)
    results, details = {}, {}
    try:
        for fleurs in languages:
            entry = manifest["languages"][fleurs]
            start = time.perf_counter()
            hypotheses, stops = [], []
            for path in paths[fleurs]:
                text, stop = run(path, entry["lang"])
                hypotheses.append(text)
                stops.append(stop)
            references = [u["reference"] for u in entry["utterances"][:len(hypotheses)]]
            results[fleurs] = score(references, hypotheses, entry["metric"])
            if args.model == "llm":
                results[fleurs]["truncated"] = sum(s != "eos" for s in stops)
            results[fleurs]["seconds"] = round(time.perf_counter() - start, 1)
            details[fleurs] = dict(zip((p.name for p in paths[fleurs]), hypotheses))
            print(json.dumps({"language": fleurs, **results[fleurs]}), flush=True)
    finally:
        asr.close()
    failures = check(results, model, manifest) if args.gate else []
    skipped = skipped_languages(results, model) if args.gate else []
    report = {"model": args.model, "repo": repo, "revision": revision,
              "dataset": DATASET, "dataset_revision": manifest["revision"],
              "per_language": {fleurs: len(paths[fleurs]) for fleurs in languages},
              "normalization": "evaluate_precisions.normalize; whitespace removed for CER-gated "
                               "languages",
              "results": results, "failures": failures, "skipped_gated_languages": skipped,
              "hypotheses": details}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for failure in failures:
        print(f"REGRESSION {failure}", file=sys.stderr)
    if skipped:
        print(f"note: gated languages not measured in this run: {skipped}", file=sys.stderr)
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="score a model per language")
    run.add_argument("--model", choices=["ctc", "llm"], required=True)
    run.add_argument("--repo", help="override the pinned model repo")
    run.add_argument("--revision", help="override the pinned model revision")
    run.add_argument("--languages", nargs="+", help="subset of manifest languages")
    run.add_argument("--per-language", type=int,
                     help="first N utterances per language (default: the model's setting)")
    run.add_argument("--manifest", type=Path, default=MANIFEST)
    run.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    run.add_argument("--audio-dir", type=Path, default=Path(".cache/fleurs"))
    run.add_argument("--cache-dir", type=Path, help="model and dataset download cache")
    run.add_argument("--gate", action="store_true", help="exit 1 if a threshold is exceeded")
    run.add_argument("--report", type=Path)
    build = sub.add_parser("build-manifest", help="maintainers: regenerate the manifest")
    build.add_argument("--revision", required=True, help="FLEURS commit to pin")
    build.add_argument("--fleurs-root", type=Path,
                       help="local data/ directory of that commit (default: download)")
    build.add_argument("--per-language", type=int, default=20)
    build.add_argument("--out", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    if args.command == "run":
        return evaluate(args)
    root = args.fleurs_root
    if root is None:
        from huggingface_hub import snapshot_download

        patterns = [f"data/{f}/test.tsv" for f in LANGUAGES]
        patterns += [f"data/{f}/audio/test.tar.gz" for f in LANGUAGES]
        root = Path(snapshot_download(DATASET, repo_type="dataset", revision=args.revision,
                                      allow_patterns=patterns)) / "data"
    manifest = build_manifest(root, args.revision, args.per_language)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {sum(len(e['utterances']) for e in manifest['languages'].values())}"
          " utterances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
