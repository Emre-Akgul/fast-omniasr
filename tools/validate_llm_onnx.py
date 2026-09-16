"""Compare raw ONNX tensors and greedy trajectories with the frozen Day-2 oracle."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from fast_omniasr.llm.onnx_backend import (
    ONNXLLMBackend,
    decoder_session,
    decoder_sessions,
    session,
)


def compare(actual, expected, label, atol=2e-4, rtol=2e-4):
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
    error = np.abs(actual - expected)
    return {
        "max_abs_error": float(error.max()),
        "max_relative_error": float((error / np.maximum(np.abs(expected), 1e-8)).max()),
    }


def cache_feed(session, token, cache):
    names = [value.name for value in session.get_inputs()]
    return dict(zip(names, [token, *cache], strict=True))


def check_outputs(actual, logits, cache, label, atol=2e-4, rtol=2e-4):
    expected = [logits.numpy(), *[value.numpy() for value in cache]]
    if len(actual) != len(expected):
        raise AssertionError("Wrong number of cache outputs")
    errors = [compare(a, b, f"{label}:{i}", atol, rtol) for i, (a, b) in enumerate(zip(actual, expected))]
    if actual[0].argmax(-1).item() != expected[0].argmax(-1).item():
        raise AssertionError(f"Greedy mismatch: {label}")
    return {
        "logits": errors[0],
        "max_key_error": max(e["max_abs_error"] for e in errors[1::2]),
        "max_value_error": max(e["max_abs_error"] for e in errors[2::2]),
        "cache_shapes": [list(a.shape) for a in actual[1:]],
        "argmax_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stage", choices=["probes", "contexts", "generation"], default="probes")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    suite = json.loads((args.source / "cached_reference.json").read_text())
    if args.stage == "contexts":
        from fast_omniasr.audio import load_audio

        model = torch.jit.load(str(args.source / "encoder.pt")).eval()
        encoder = session(args.directory / "encoder.onnx", args.device)
        arrays, reports = {}, []
        with torch.inference_mode():
            for i, clip in enumerate(suite["clips"]):
                audio = load_audio(args.source / clip["audio"])
                assert (
                    hashlib.sha256(audio.tobytes()).hexdigest() == clip["normalized_audio_sha256"]
                )
                expected = model(torch.from_numpy(audio)).numpy()
                actual = encoder.run(None, {"audio": audio})[0]
                assert actual.shape == expected.shape and np.isfinite(actual).all()
                error = np.abs(actual - expected)
                scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1.0)
                scaled = float((error / scale).max())
                relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(expected))
                if scaled > 1e-3 or relative_l2 > 1e-4:
                    raise AssertionError(
                        f"Encoder error on {clip['audio']}: {scaled}, {relative_l2}"
                    )
                arrays[f"ts_{i}"] = expected
                arrays[f"ort_{i}"] = actual
                report = {
                    "audio": clip["audio"],
                    "shape": list(actual.shape),
                    "max_abs_error": float(error.max()),
                    "max_relative_error": float((error / np.maximum(np.abs(expected), 1e-8)).max()),
                    "max_channel_scaled_error": scaled,
                    "relative_l2_error": relative_l2,
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
        np.savez(args.directory / f"contexts_{args.device}.npz", **arrays)
        (args.directory / f"contexts_{args.device}.json").write_text(
            json.dumps(reports, indent=2) + "\n"
        )
        return
    model = torch.jit.load(str(args.source / "decoder_cached.pt")).eval()
    if args.device == "cpu" and args.stage != "generation":
        prefill, step, keepalive = decoder_sessions(args.directory, args.device)
        backend = None
    elif args.device == "cpu":
        prefill = step = None
        keepalive = []
        backend = None
    else:
        prefill = step = None
        keepalive = []
        backend = ONNXLLMBackend(args.directory, args.device)
    rng = np.random.default_rng(19)
    bos = torch.tensor([[0]], dtype=torch.int64)
    if args.stage == "generation":
        if not (args.directory / f"decoder_probes_{args.device}.json").exists():
            raise ValueError("Run the independent decoder probes first")
        arrays = np.load(args.directory / f"contexts_{args.device}.npz")
        config = json.loads((args.source / "config.json").read_text())
        reports = []
        with torch.inference_mode():
            for i, clip in enumerate(suite["clips"]):
                context_ts = torch.from_numpy(arrays[f"ts_{i}"])
                context_ort = arrays[f"ort_{i}"]
                logits, cache = model.prefill(context_ts, bos)
                if backend is None:
                    prefill, prefill_keepalive = decoder_session(
                        args.directory, "decoder_prefill.onnx"
                    )
                    actual = prefill.run(None, {"context": context_ort, "bos": bos.numpy()})
                    del prefill, prefill_keepalive
                    gc.collect()
                    step, step_keepalive = decoder_session(
                        args.directory, "decoder_step.onnx"
                    )
                else:
                    backend.close()
                    actual_logits, actual_cache = backend.prefill(context_ort, bos.numpy())
                    actual = [actual_logits, *actual_cache]
                records, tokens = [], []
                for index in range(512):
                    record = check_outputs(
                        actual,
                        logits,
                        cache,
                        f"{clip['audio']}:{index}",
                        atol=2e-3,
                        rtol=2e-3,
                    )
                    token = int(logits.argmax(-1))
                    record.update({"step": index, "token": token})
                    records.append(record)
                    if token == config["eos_idx"]:
                        break
                    tokens.append(token)
                    ids = torch.tensor([[token]], dtype=torch.int64)
                    logits, cache = model.decode_step(ids, cache)
                    if backend is None:
                        actual = step.run(None, cache_feed(step, ids.numpy(), actual[1:]))
                    else:
                        actual_logits, actual_cache = backend.decode_step(ids.numpy(), actual[1:])
                        actual = [actual_logits, *actual_cache]
                    if (index + 1) % 10 == 0:
                        print(f"{clip['audio']}: matched {index + 1} steps", flush=True)
                else:
                    raise AssertionError("Did not reach EOS")
                assert tokens == clip["token_ids"]
                report = {
                    "audio": clip["audio"],
                    "token_ids": tokens,
                    "stop_reason": "eos",
                    "decode_steps_including_eos": len(records),
                    "steps": records,
                }
                reports.append(report)
                print(f"{clip['audio']}: matched EOS at {len(records)} steps", flush=True)
                if backend is None:
                    del step, step_keepalive
                    gc.collect()
        (args.directory / f"generation_{args.device}.json").write_text(
            json.dumps(
                {"device": args.device, "atol": 2e-3, "rtol": 2e-3, "clips": reports},
                indent=2,
            )
            + "\n"
        )
        return
    probes = []
    with torch.inference_mode():
        for length in [10, 50, 150, 300]:
            context = torch.from_numpy(rng.normal(size=(1, length - 1, 4096)).astype(np.float32))
            logits, cache = model.prefill(context, bos)
            if backend is None:
                actual = prefill.run(None, {"context": context.numpy(), "bos": bos.numpy()})
            else:
                backend.close()
                actual_logits, actual_cache = backend.prefill(context.numpy(), bos.numpy())
                actual = [actual_logits, *actual_cache]
            prefill_report = check_outputs(actual, logits, cache, f"prefill-{length}")
            token = logits.argmax(-1).reshape(1, 1)
            next_logits, next_cache = model.decode_step(token, cache)
            if backend is None:
                stepped = step.run(None, cache_feed(step, token.numpy(), actual[1:]))
            else:
                stepped_logits, stepped_cache = backend.decode_step(token.numpy(), actual[1:])
                stepped = [stepped_logits, *stepped_cache]
            step_report = check_outputs(stepped, next_logits, next_cache, f"step-{length}")
            report = {"cache_length": length, "prefill": prefill_report, "step": step_report}
            probes.append(report)
            print(json.dumps(report), flush=True)
    del prefill, step, keepalive, model
    gc.collect()
    (args.directory / f"decoder_probes_{args.device}.json").write_text(
        json.dumps({"device": args.device, "atol": 2e-4, "rtol": 2e-4, "probes": probes}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
