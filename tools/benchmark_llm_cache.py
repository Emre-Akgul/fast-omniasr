"""Compare decoder-only scaling on identical forced tokens (EOS does not stop timing)."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from fast_omniasr.audio import load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(4)
    config = json.loads((args.directory / "config.json").read_text())
    reference = json.loads((args.directory / "reference.json").read_text())
    audio = args.audio or args.directory / reference["audio"]
    encoder = torch.jit.load(str(args.directory / "encoder.pt")).eval()
    rows = []
    with torch.inference_mode():
        context = encoder(torch.from_numpy(load_audio(audio)))
        del encoder
        gc.collect()
        # Repeated fixture tokens isolate length scaling from transcript differences.
        sequence = reference["token_ids"] + [config["eos_idx"]]
        ids = torch.tensor([[config["bos_idx"]] + (sequence * 101)[:100]])
        if context.shape[1] + 100 > config["max_seq_len"]:
            raise ValueError("Benchmark exceeds model context capacity")

        def run(decoder, use_cache, count):
            times = []
            cache = None
            for i in range(count):
                start = time.perf_counter()
                if use_cache:
                    if i == 0:
                        _, cache = decoder.prefill(context, ids[:, :1])
                    else:
                        _, cache = decoder.decode_step(ids[:, i : i + 1], cache)
                else:
                    decoder(context, ids[:, : i + 1])
                times.append(time.perf_counter() - start)
                if (i + 1) % 25 == 0:
                    print(
                        f"Timing {'cached' if use_cache else 'full-prefix'}: step {i + 1}",
                        flush=True,
                    )
            return times

        # Load modes sequentially to avoid keeping two decoder weight sets in RAM.
        measurements = {}
        for mode, filename in ((False, "decoder.pt"), (True, "decoder_cached.pt")):
            decoder = torch.jit.load(str(args.directory / filename)).eval()
            run(decoder, mode, 10)
            runs = []
            for repeat in range(args.repeats):
                runs.append(run(decoder, mode, 100))
                print(f"Measured {filename}: 100 steps, run {repeat + 1}", flush=True)
            for count in (10, 25, 50, 100):
                measurements[mode, count] = [r[:count] for r in runs]
            del decoder
            gc.collect()
        for count in (10, 25, 50, 100):
            row = {"output_steps": count}
            for mode, name in ((False, "full_prefix"), (True, "cached")):
                runs = measurements[mode, count]
                row[name] = {
                    "total_seconds": statistics.median(sum(r) for r in runs),
                    "first_step_ms": 1000 * statistics.median(r[0] for r in runs),
                    "last_5_steps_mean_ms": 1000
                    * statistics.median(statistics.mean(r[-5:]) for r in runs),
                    "runs_seconds": [sum(r) for r in runs],
                }
            row["speedup"] = row["full_prefix"]["total_seconds"] / row["cached"]["total_seconds"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "torch_version": torch.__version__,
        "threads": 4,
        "device": "cpu",
        "audio": str(audio),
        "audio_context_positions": context.shape[1],
        "repeats": args.repeats,
        "encoder_timed": False,
        "protocol": "Identical forced tokens; includes prefill; ignores EOS; median warm runs; 10/25/50 are prefixes of each 100-step run",
        "measurements": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
