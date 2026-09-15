"""Add explicit-cache graphs to an existing full-prefix export, validating three-way parity."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from llm_cache import ExplicitDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional original checkpoint; otherwise reuse baseline weights",
    )
    parser.add_argument("--audio", type=Path, action="append", default=[])
    args = parser.parse_args()
    import omnilingual_asr  # noqa: F401
    from fairseq2.models.transformer import AttentionState
    from fairseq2.nn import BatchLayout, IncrementalStateBag
    from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub

    from fast_omniasr.audio import load_audio

    torch.set_num_threads(4)
    config = json.loads((args.directory / "config.json").read_text())
    if (config.get("format_version"), config.get("model")) != (1, "omniASR_LLM_300M_v2"):
        raise ValueError("Unsupported LLM artifact format or model")
    reference = json.loads((args.directory / "reference.json").read_text())
    clips = [args.directory / reference["audio"], *args.audio]
    temporary = tempfile.TemporaryDirectory(prefix="llm-oracle-")
    oracle_dir = Path(temporary.name)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("llm_baseline_oracle.py")),
            str(args.directory),
            str(oracle_dir),
            *map(str, clips),
        ],
        check=True,
    )
    hub = get_wav2vec2_llama_model_hub()
    if args.checkpoint:
        model = hub.load_custom_model(
            args.checkpoint,
            hub.get_model_config(config["model"]),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
        )
        model.eval().requires_grad_(False)
    else:
        # Recreate real fairseq2 modules, sharing the baseline's exact parameters.
        # Meta construction avoids allocating a second set of decoder weights.
        from fairseq2.models.llama import LLaMAFactory

        baseline = torch.jit.load(str(args.directory / "decoder.pt")).eval()
        factory = LLaMAFactory(hub.get_model_config(config["model"]).llama_config)
        with torch.device("meta"):
            decoder = factory.create_decoder()
        weights = {
            k: v
            for k, v in baseline.decoder.state_dict().items()
            if not k.endswith("pos_encoder.freqs")
        }
        decoder.load_state_dict(weights, strict=True, assign=True)
        for layer, original in zip(decoder.layers, baseline.decoder.layers.children()):
            position = factory.create_position_encoder()
            torch.testing.assert_close(
                position.freqs, original.self_attn.pos_encoder.freqs, rtol=0, atol=0
            )
            layer.self_attn.pos_encoder = position
        decoder.eval().requires_grad_(False)
        model = SimpleNamespace(
            text_frontend=baseline.embedding, llama_decoder=decoder, final_proj=baseline.proj
        )
    print("Reference decoder loaded", flush=True)
    explicit = ExplicitDecoder(model).eval()
    bos = torch.tensor([[config["bos_idx"]]])
    reports = []
    with torch.inference_mode():
        context = torch.load(oracle_dir / "0.pt", weights_only=True)["context"]
        _, cache = explicit.prefill(context, bos)
        graph = torch.jit.trace_module(
            explicit,
            {"prefill": (context, bos), "decode_step": (bos, cache)},
            check_trace=False,
            strict=False,
        )
        for clip_index, audio in enumerate(clips):
            waveform = load_audio(audio)
            if waveform.shape[1] > 480000:
                raise ValueError("Clips must be <=30 seconds")
            oracle = torch.load(oracle_dir / f"{clip_index}.pt", weights_only=True)
            context = oracle["context"]
            state = IncrementalStateBag(max_num_steps=config["max_seq_len"])
            model.llama_decoder(
                context,
                BatchLayout(context.shape[:2], seq_lens=None, device=context.device),
                state_bag=state,
            )
            state.increment_step_nr(context.shape[1])
            actual, cache = graph.prefill(context, bos)
            ids = [config["bos_idx"]]
            max_error = 0.0
            max_cache_error = 0.0
            budget = min(512, config["max_seq_len"] - context.shape[1] - 3)
            for step in range(budget):
                token = torch.tensor([[ids[-1]]])
                x = model.text_frontend(token)
                out = model.llama_decoder(
                    x, BatchLayout(x.shape[:2], seq_lens=None, device=x.device), state_bag=state
                )
                state.increment_step_nr(1)
                expected = model.final_proj(out[:, -1])
                full = oracle["logits"][step]
                if step:
                    actual, cache = graph.decode_step(token, cache)
                for logits in (expected, full, actual):
                    assert torch.isfinite(logits).all()
                tokens = [int(logits.argmax(-1)) for logits in (expected, full, actual)]
                assert len(set(tokens)) == 1, (str(audio), step, tokens)
                max_error = max(max_error, float((expected - actual).abs().max()))
                # Check the actual state, not only its effect on argmax.
                for i, layer in enumerate(model.llama_decoder.layers):
                    ref_k, ref_v = state.maybe_get_state(layer.self_attn, AttentionState).get()
                    for wanted, got in zip((ref_k, ref_v), cache[2 * i : 2 * i + 2]):
                        assert got.shape == wanted.shape
                        torch.testing.assert_close(got, wanted, atol=2e-4, rtol=2e-4)
                        max_cache_error = max(max_cache_error, float((wanted - got).abs().max()))
                if tokens[0] == config["eos_idx"]:
                    break
                ids.append(tokens[0])
            else:
                raise AssertionError(f"Did not reach EOS: {audio}")
            if clip_index == 0:
                assert ids[1:] == reference["token_ids"]
            bundled = (
                reference["audio"] if clip_index == 0 else f"parity-{clip_index}{audio.suffix}"
            )
            if clip_index:
                shutil.copyfile(audio, args.directory / bundled)
            report = {
                "audio": bundled,
                "normalized_audio_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                "token_ids": ids[1:],
                "decode_steps_including_eos": step + 1,
                "prefill_positions": context.shape[1] + 1,
                "final_cache_positions": cache[0].shape[1],
                "stop_reason": "eos",
                "max_logit_abs_error": max_error,
                "max_cache_abs_error": max_cache_error,
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
        graph.save(str(args.directory / "decoder_cached.pt"))
    report = {
        "cache_layout": "alternating K,V per layer; [batch, sequence, kv_heads, head_dim]",
        "layers": len(explicit.layers),
        "kv_heads": explicit.kv_heads,
        "head_dim": explicit.head_dim,
        "torch_version": torch.__version__,
        "reference_weights": "checkpoint" if args.checkpoint else "preserved decoder.pt",
        "fairseq2_version": __import__("fairseq2").__version__,
        "clips": reports,
    }
    (args.directory / "cached_reference.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.cleanup()


if __name__ == "__main__":
    main()
