import argparse, io, json, time
from datetime import datetime, timezone

import requests
import soundfile as sf
from datasets import Audio, load_dataset

from fast_omniasr import OmniASR
from metrics import compare, summarize

TTC_URL = "http://localhost:4395/api/v1/text_to_commands"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = OmniASR(args.engine, args.tokenizer, backend="tensorrt")

    ds = load_dataset("TartarusXXX/Drone-April-1-TTC", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda r: r["evaluation"] == "approved")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Loaded {len(ds)} examples")

    results = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        filename = audio.get("path") or "audio.wav"
        raw_ref = row.get("commands") or row.get("command")
        reference = json.loads(raw_ref) if raw_ref else []

        result = {"filename": filename, "reference": reference}
        t0 = time.perf_counter()
        try:
            samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            t_stt0 = time.perf_counter()
            text = model.transcribe_numpy(samples, sample_rate=sr).text
            t_stt1 = time.perf_counter()
            result["predicted_text"] = text
            result["stt_time_s"] = t_stt1 - t_stt0
        except Exception as exc:
            result["error"] = str(exc)
            result["comparison"] = {"loose_match": False, "strict_match": False}
            result["total_time_s"] = time.perf_counter() - t0
            results.append(result)
            print(f"[{i+1}/{len(ds)}] {filename} loose={result['comparison']['loose_match']} strict={result['comparison']['strict_match']}")
            continue

        try:
            t_ttc0 = time.perf_counter()
            resp = requests.post(TTC_URL, json={"text": text, "async_task_id": "trt-eval"}, timeout=60)
            resp.raise_for_status()
            commands = resp.json()["response"]["commands"]
            t_ttc1 = time.perf_counter()
            result["predicted_commands"] = commands
            result["ttc_time_s"] = t_ttc1 - t_ttc0
            result["comparison"] = compare(commands, reference)
        except Exception as exc:
            result["error"] = str(exc)
            result["comparison"] = {"loose_match": False, "strict_match": False}
        result["total_time_s"] = time.perf_counter() - t0
        results.append(result)
        print(f"[{i+1}/{len(ds)}] {filename} loose={result['comparison']['loose_match']} strict={result['comparison']['strict_match']}")

    summary = summarize(results)
    print()
    print(json.dumps(summary, indent=2))
    out_path = f"results_{args.run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
