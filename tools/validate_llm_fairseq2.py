"""Compare ONNX LLM-ASR logits directly with fairseq2 incremental greedy decoding.

Two separate processes keep the comparison honest:

  reference: fairseq2 + omnilingual_asr, CPU float32. Records the logits of every
             greedy step through EOS for each clip.
  compare:   PyTorch, fairseq2 and omnilingual_asr imports are blocked. Loads
             published (--repo, via from_pretrained) or local (--directory) ONNX
             assets, replays the reference tokens, and compares every step's
             logits, then checks the public transcribe() output end to end.

  python tools/validate_llm_fairseq2.py reference --model omniASR_LLM_1B_v2 \\
      --checkpoint omniASR-LLM-1B-v2.pt --output ref/ a.wav b.wav --lang eng_Latn --lang none
  python tools/validate_llm_fairseq2.py compare ref/ --repo EmreAkgul/omniASR-LLM-1B-v2-ONNX
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

THREADS = int(os.environ.get("FAST_OMNIASR_THREADS", "4"))


def reference(args):
    import omnilingual_asr
    import torch
    from fairseq2.data.tokenizers import load_tokenizer
    from fairseq2.datasets.batch import Seq2SeqBatch
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout, IncrementalStateBag
    from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub

    from fast_omniasr.audio import load_audio
    from fast_omniasr.llm.models import LLM_MODEL_TOKENIZERS

    torch.set_num_threads(THREADS)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.checkpoint:
        hub = get_wav2vec2_llama_model_hub()
        model = hub.load_custom_model(
            args.checkpoint,
            hub.get_model_config(args.model),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
        )
    else:
        model = load_model(args.model, device=torch.device("cpu"), dtype=torch.float32)
    model.eval().requires_grad_(False)
    text_decoder = load_tokenizer(LLM_MODEL_TOKENIZERS[args.model]).create_decoder(
        skip_special_tokens=True
    )
    bos = model.target_vocab_info.bos_idx
    eos = model.target_vocab_info.eos_idx
    langs = args.lang or ["none"] * len(args.audio)
    if len(langs) != len(args.audio):
        raise ValueError("Pass one --lang per clip (use 'none' for no conditioning)")
    clips = []
    with torch.inference_mode():
        for index, (audio, lang) in enumerate(zip(args.audio, langs)):
            lang = None if lang == "none" else lang
            if lang is not None and lang.lower() not in model.lang_mapping:
                raise ValueError(f"Unknown language code {lang}")
            waveform = torch.from_numpy(load_audio(audio))
            if waveform.shape[1] > 480000:
                raise ValueError("Clips must be <=30 seconds")
            batch = Seq2SeqBatch(
                source_seqs=waveform,
                source_seq_lens=[waveform.shape[1]],
                target_seqs=torch.tensor([[bos]]),
                target_seq_lens=[1],
                example={} if lang is None else {"lang": [lang]},
            )
            contexts, lengths, _ = model(batch, return_decoder_inputs=True)
            # Upstream prefill: everything before BOS, then BOS is the first step.
            prefix = contexts[0][:, : lengths[0][0] - 1]
            state = IncrementalStateBag(max_num_steps=model.max_generation_length)
            model.llama_decoder(
                prefix,
                BatchLayout(prefix.shape[:2], seq_lens=None, device=prefix.device),
                state_bag=state,
            )
            state.increment_step_nr(prefix.shape[1])
            budget = min(args.max_new_tokens, model.max_generation_length - prefix.shape[1] - 3)
            ids, logits = [bos], []
            for _ in range(budget):
                x = model.text_frontend(torch.tensor([[ids[-1]]]))
                out = model.llama_decoder(
                    x, BatchLayout(x.shape[:2], seq_lens=None, device=x.device), state_bag=state
                )
                state.increment_step_nr(1)
                step_logits = model.final_proj(out[:, -1])
                if not torch.isfinite(step_logits).all():
                    raise AssertionError(f"Non-finite fairseq2 logits for {audio}")
                logits.append(step_logits[0].numpy())
                token = int(step_logits.argmax(-1))
                if token == eos:
                    break
                ids.append(token)
            else:
                raise AssertionError(f"fairseq2 did not reach EOS for {audio}")
            name = f"{index}.npy" if lang is None else f"{index}-{lang}.npy"
            np.save(args.output / name, np.stack(logits))
            clip = {
                "audio": str(Path(audio).resolve()),
                "lang": lang,
                "normalized_audio_sha256": hashlib.sha256(waveform.numpy().tobytes()).hexdigest(),
                "token_ids": ids[1:],
                "text": text_decoder(torch.tensor(ids[1:], dtype=torch.int64)),
                "decode_steps_including_eos": len(logits),
                "logits": name,
            }
            clips.append(clip)
            print(json.dumps({k: clip[k] for k in ("audio", "lang", "text")}), flush=True)
    report = {
        "model": args.model,
        "bos_idx": bos,
        "eos_idx": eos,
        "dtype": "float32",
        "device": "cpu",
        "decoding": "fairseq2 incremental greedy",
        "torch_version": torch.__version__,
        "fairseq2_version": __import__("fairseq2").__version__,
        "omnilingual_asr_version": getattr(omnilingual_asr, "__version__", "unknown"),
        "clips": clips,
    }
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(args):
    from validate_llm import BlockReferenceImports

    sys.meta_path.insert(0, BlockReferenceImports(forbid_torch=True))
    from fast_omniasr.audio import load_audio
    from fast_omniasr.llm import OmniASRLLM

    suite = json.loads((args.reference / "reference.json").read_text())
    if args.repo:
        model = OmniASRLLM.from_pretrained(args.repo, device=args.device, threads=THREADS)
    else:
        model = OmniASRLLM(args.directory, backend="onnx", device=args.device, threads=THREADS)
    if model.config["model"] != suite["model"]:
        raise AssertionError(f"Artifact is {model.config['model']}, reference is {suite['model']}")
    reports = []
    for clip in suite["clips"]:
        audio = load_audio(clip["audio"])
        if hashlib.sha256(audio.tobytes()).hexdigest() != clip["normalized_audio_sha256"]:
            raise AssertionError(f"Audio differs from the reference: {clip['audio']}")
        expected = np.load(args.reference / clip["logits"])
        model.onnx.close()
        context = model.onnx.encode(audio)
        if clip.get("lang"):
            context = np.array(context, copy=True)
            context[:, -1] = model.language_embedding(clip["lang"])
        logits, cache = model.onnx.prefill(context, np.array([[suite["bos_idx"]]], np.int64))
        errors, mismatches = [], []
        for step, wanted in enumerate(expected):
            actual = logits[0]
            if not np.isfinite(actual).all():
                raise AssertionError(f"Non-finite ONNX logits at step {step}")
            errors.append(float(np.abs(actual - wanted).max()))
            if int(actual.argmax()) != int(wanted.argmax()):
                mismatches.append(step)
            if step + 1 < len(expected):
                token = np.array([[clip["token_ids"][step]]], np.int64)
                logits, cache = model.onnx.decode_step(token, cache)
        model.onnx.close()
        result = model.transcribe(audio, max_new_tokens=512, lang=clip.get("lang"))
        report = {
            "audio": Path(clip["audio"]).name,
            "lang": clip.get("lang"),
            "decode_steps_including_eos": len(expected),
            "max_logit_abs_error": max(errors),
            "first_step_logit_abs_error": errors[0],
            "argmax_mismatch_steps": mismatches,
            "tokens_equal": result.token_ids == clip["token_ids"],
            "text_equal": result.text == clip["text"],
            "stop_reason": result.stop_reason,
            "text": result.text,
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    summary = {
        "model": suite["model"],
        "source": args.repo or str(args.directory),
        "device": args.device,
        "atol": args.atol,
        "max_logit_abs_error": max(r["max_logit_abs_error"] for r in reports),
        "total_steps": sum(r["decode_steps_including_eos"] for r in reports),
        "passed": all(
            not r["argmax_mismatch_steps"]
            and r["tokens_equal"]
            and r["text_equal"]
            and r["stop_reason"] == "eos"
            and r["max_logit_abs_error"] <= args.atol
            for r in reports
        ),
        "clips": reports,
    }
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}), flush=True)
    if not summary["passed"]:
        raise SystemExit("ONNX output does not match fairseq2")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--model", required=True)
    ref.add_argument("--checkpoint", type=Path)
    ref.add_argument("--output", type=Path, required=True)
    ref.add_argument("--max-new-tokens", type=int, default=512)
    ref.add_argument(
        "--lang", action="append", help="Per-clip language code, in clip order; 'none' to omit"
    )
    ref.add_argument("audio", type=Path, nargs="+")
    cmp = commands.add_parser("compare")
    cmp.add_argument("reference", type=Path)
    source = cmp.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path)
    source.add_argument("--repo")
    cmp.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    cmp.add_argument("--atol", type=float, default=1e-2)
    cmp.add_argument("--report", type=Path)
    args = parser.parse_args()
    reference(args) if args.command == "reference" else compare(args)


if __name__ == "__main__":
    main()
