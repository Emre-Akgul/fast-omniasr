"""Export a separate full-prefix LLM runtime. fairseq2 is used only here."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL = "omniASR_LLM_300M_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audio", type=Path, required=True, help="Mono 16 kHz parity clip (up to 30 seconds)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    import omnilingual_asr
    import torch
    from fairseq2.datasets.batch import Seq2SeqBatch
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout
    from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub

    from export_onnx import prepare_tokenizer
    from fast_omniasr.audio import load_audio

    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError("Use a new output directory to avoid mixing artifact versions")
    waveform = torch.from_numpy(load_audio(args.audio))
    if waveform.shape[1] > 480000 or args.max_new_tokens < 1:
        raise ValueError("Use a clip <=30 seconds and a positive generation limit")
    if args.checkpoint:
        hub = get_wav2vec2_llama_model_hub()
        model = hub.load_custom_model(
            args.checkpoint,
            hub.get_model_config(MODEL),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
        )
    else:
        model = load_model(MODEL, device=torch.device("cpu"), dtype=torch.float32)
    model.eval().requires_grad_(False)
    if model.encoder_stacking != 1 or model.streaming_config.is_streaming:
        raise ValueError(
            "The experimental LLM exporter supports only non-streaming encoder_stacking=1 models"
        )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = model.encoder_frontend
            self.encoder = model.encoder
            self.proj = model.encoder_proj
            self.register_buffer(
                "suffix",
                torch.cat(
                    [
                        model.text_frontend(torch.tensor([[model.special_tokens.lid_marker]])),
                        model.lang_embeddings(torch.tensor([[0]])),
                    ],
                    dim=1,
                ),
            )

        def forward(self, audio):
            layout = BatchLayout(audio.shape, seq_lens=None, device=audio.device)
            x, layout, _ = self.frontend.extract_features(audio, layout)
            x, _ = self.frontend.process_features(x, layout, None)
            x = self.proj(self.encoder(x, layout))
            return torch.cat([x, self.suffix], dim=1)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = model.text_frontend
            self.decoder = model.llama_decoder
            self.proj = model.final_proj

        def forward(self, context, tokens):
            x = torch.cat([context, self.embedding(tokens)], dim=1)
            layout = BatchLayout(x.shape[:2], seq_lens=None, device=x.device)
            return self.proj(self.decoder(x, layout)[:, -1])

    encoder, decoder = Encoder().eval(), Decoder().eval()
    bos = model.target_vocab_info.bos_idx
    eos = model.target_vocab_info.eos_idx
    with torch.inference_mode():
        print("Checking reference syntax", flush=True)
        batch = Seq2SeqBatch(
            source_seqs=waveform,
            source_seq_lens=[waveform.shape[1]],
            target_seqs=torch.tensor([[bos]]),
            target_seq_lens=[1],
            example={},
        )
        contexts, lengths, _ = model(batch, return_decoder_inputs=True)
        context = encoder(waveform)
        prefix = torch.cat([context, model.text_frontend(torch.tensor([[bos]]))], dim=1)
        torch.testing.assert_close(prefix, contexts[0][:, : lengths[0][0]])
        print("Tracing encoder and decoder", flush=True)
        # fairseq2 infers is_causal with a Python shape comparison, which becomes
        # a Tensor under tracing. Batch-one full-prefix graphs have fixed causal
        # semantics; retain the original modules for the incremental reference.
        from fairseq2.models.transformer import CausalAttentionBias, TorchSDPA

        class TraceSDPA(torch.nn.Module):
            def __init__(self, causal):
                super().__init__()
                self.causal = causal

            def forward(self, q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=self.causal,
                )
                return out.transpose(1, 2), None

        originals = []
        for module in model.modules():
            if hasattr(module, "sdpa") and isinstance(module.sdpa, TorchSDPA):
                originals.append((module, module.sdpa))
                module.sdpa = TraceSDPA(isinstance(module.sdpa.bias, CausalAttentionBias))
        enc = torch.jit.trace(encoder, (waveform,), check_trace=False)
        dec = torch.jit.trace(decoder, (context, torch.tensor([[bos]])), check_trace=False)
        for module, original in originals:
            module.sdpa = original
        # Explicitly exercise unseen lengths; tracing Python lengths can silently freeze shapes.
        encoder_probe_samples = [400, 8000, 16000, 32000, 64000, 480000]
        for n in encoder_probe_samples:
            print(f"Checking encoder parity at {n} samples", flush=True)
            probe = torch.randn(1, n)
            torch.testing.assert_close(enc(probe), encoder(probe))
        for n in [1, 2, 5]:
            ids = torch.full((1, n), bos, dtype=torch.int64)
            short_context = context[:, : min(7, context.shape[1])]
            torch.testing.assert_close(dec(short_context, ids), decoder(short_context, ids))
        print("Comparing full-prefix generation against fairseq2 incremental decoding", flush=True)
        from fairseq2.nn import IncrementalStateBag

        state = IncrementalStateBag(max_num_steps=model.max_generation_length)
        exported_context = enc(waveform)
        # Match upstream prefill: consume everything except BOS, then step BOS.
        reference_context = contexts[0][:, : lengths[0][0] - 1]
        model.llama_decoder(
            reference_context,
            BatchLayout(reference_context.shape[:2], seq_lens=None, device=waveform.device),
            state_bag=state,
        )
        state.increment_step_nr(reference_context.shape[1])
        ids = [bos]
        reference_ids = []
        first_error = None
        max_logit_error = 0.0
        decode_steps = 0
        budget = min(args.max_new_tokens, model.max_generation_length - prefix.shape[1] - 2)
        if budget < 1:
            raise ValueError("Audio prefix exceeds decoder context capacity")
        reason = "max_new_tokens" if budget == args.max_new_tokens else "context_limit"
        for step in range(budget):
            x = model.text_frontend(torch.tensor([[ids[-1]]]))
            layout = BatchLayout(x.shape[:2], seq_lens=None, device=x.device)
            out = model.llama_decoder(x, layout, state_bag=state)
            state.increment_step_nr(x.shape[1])
            reference = model.final_proj(out[:, -1])
            actual = dec(exported_context, torch.tensor([ids]))
            if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
                raise AssertionError(f"Non-finite logits at generation step {step}")
            step_error = float((reference - actual).abs().max())
            max_logit_error = max(max_logit_error, step_error)
            decode_steps += 1
            if first_error is None:
                first_error = step_error
            expected_token, actual_token = int(reference.argmax(-1)), int(actual.argmax(-1))
            if expected_token != actual_token:
                raise AssertionError(
                    f"Token mismatch at step {step}: {expected_token} != {actual_token}"
                )
            if expected_token == eos:
                reason = "eos"
                break
            if step % 25 == 0:
                print(f"Matched generation step {step}", flush=True)
            reference_ids.append(expected_token)
            ids.append(expected_token)
        if reason != "eos":
            raise AssertionError("Only a matching prefix was obtained; increase --max-new-tokens")
        args.output.mkdir(parents=True)
        enc.save(str(args.output / "encoder.pt"))
        dec.save(str(args.output / "decoder.pt"))
        # The upstream 300m_v2 config can report a stale target vocabulary size.
        # Tokenizer compatibility is determined by output logits, while prefix
        # markers must preserve the reference model's actual generation behavior.
        prepare_tokenizer(MODEL, args.output / "tokenizer.model", model.final_proj.output_dim)
        config = {
            "format_version": 1,
            "model": MODEL,
            "bos_idx": bos,
            "eos_idx": eos,
            "pad_idx": model.target_vocab_info.pad_idx,
            "vocab_size": model.final_proj.output_dim,
            "lid_marker_idx": model.special_tokens.lid_marker,
            "reference_target_vocab_size": model.target_vocab_info.size,
            "max_seq_len": model.max_generation_length,
            "dtype": "float32",
            "fairseq2_version": __import__("fairseq2").__version__,
        }
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        from fairseq2.data.tokenizers import load_tokenizer

        from fast_omniasr.tokenizer import Tokenizer

        reference_text = load_tokenizer("omniASR_tokenizer_written_v2").create_decoder(
            skip_special_tokens=True
        )(torch.tensor(reference_ids, dtype=torch.int64))
        standalone_text = Tokenizer(args.output / "tokenizer.model").decode(reference_ids)
        if reference_text != standalone_text:
            raise AssertionError("Standalone tokenizer text differs from fairseq2")
        parity_name = f"parity{args.audio.suffix.lower()}"
        shutil.copyfile(args.audio, args.output / parity_name)
        report = {
            "token_ids": reference_ids,
            "text": reference_text,
            "stop_reason": reason,
            "first_step_max_abs_error": first_error,
            "max_logit_abs_error": max_logit_error,
            "decode_steps_including_eos": decode_steps,
            "encoder_probe_samples": encoder_probe_samples,
            "audio": parity_name,
            "normalized_audio_sha256": hashlib.sha256(waveform.numpy().tobytes()).hexdigest(),
            "torch_version": torch.__version__,
            "fairseq2_version": __import__("fairseq2").__version__,
            "omnilingual_asr_version": getattr(omnilingual_asr, "__version__", "unknown"),
            "max_new_tokens": args.max_new_tokens,
        }
        (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
