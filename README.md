# fast-omniasr

A standalone inference runtime for [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr). Runs all eight released CTC models (`300M`, `1B`, `3B`, and `7B`, both original and v2) on ONNX Runtime or TensorRT — no PyTorch, fairseq2 or Transformers on the inference path.

## Benchmark

30-second clip, RTX 3060 Laptop GPU, through the public `OmniASR` API (`benchmarks/benchmark.py`, warm median of 10 runs, includes file load, normalization, inference and decoding):

| Backend | Latency | Exact transcripts vs FP32 (40-clip FLEURS sample) |
|---|---:|---:|
| ONNX Runtime CUDA (FP32) | 358 ms | reference |
| TensorRT FP32 | 374 ms | 40/40 |
| TensorRT FP16 | **76 ms** | 34/40 |

**TensorRT FP16 is experimental and not recognition-equivalent to FP32.** See [Performance and accuracy](#performance-and-accuracy) for WER, tolerance and sample-size caveats.

<details>
<summary>Forward-only research harness numbers</summary>

The same clip measured by the original exploration harness (H2D + inference + D2H only, no file load/decode, so not directly comparable to the table above): fairseq2 FP16 137.6 ms / 1,822 MiB, TensorRT FP16 79.8 ms / 1,380 MiB, TensorRT FP32 403.5 ms / 2,096 MiB. The harness's FP16 figure and the public API's closely agree (79.8 vs. 76.4 ms), which is what justifies quoting the public number as the headline.
</details>

## Installation

```bash
python -m pip install "fast-omniasr[onnx,hub]"  # ONNX Runtime CPU + from_pretrained()
python -m pip install "fast-omniasr[cuda]"      # ONNX Runtime GPU (device="cuda")
python -m pip install "fast-omniasr[tensorrt]"  # cuda-python; also requires a separate TensorRT install
```

Automatic CPU fallback is disabled — the requested provider/backend must actually initialize. For a source checkout instead (e.g. to work on the library itself), use `pip install -e ".[onnx,hub]"` from the repo root.

## Usage

```python
from fast_omniasr import OmniASR

model = OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="onnx")
print(model.transcribe("speech.wav").text)
```

`from_pretrained` downloads `model.onnx`/`tokenizer.model` via `huggingface_hub` (requires the `hub` extra), verifies each against the sha256 in the repo's `config.json`, and caches them locally — later calls (even fully offline, e.g. `HF_HUB_OFFLINE=1`) reuse the cache.

For the fast path with no manual export or `build_tensorrt.py` call, pass `backend="tensorrt"` and an explicit `precision` (no "auto" — see [Performance and accuracy](#performance-and-accuracy) before choosing `"fp16"`):

```python
model = OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="tensorrt", precision="fp16")
```

The first call downloads the ONNX asset, builds a local engine (this takes a while — TensorRT engine builds are not fast), and caches it under `~/.cache/fast-omniasr/tensorrt/<key>/model.engine`, keyed by the ONNX content hash, precision, profile, and local TensorRT version/GPU identity (engines aren't portable across any of those). Later calls with the same key reuse the cached engine instead of rebuilding; if any of those fields change, a new engine is built rather than reusing an incompatible one. `precision="fp16"` warns once per process about the accuracy caveat.

Or supply local assets directly (no download, no `hub` extra needed):

```python
model = OmniASR("dynamic.onnx", "omniASR_tokenizer_written_v2.model")
result = model.transcribe_numpy(waveform, sample_rate=16000)  # or from a NumPy waveform
```

Audio must be mono, 16 kHz, at least 400 samples.

<details>
<summary>TensorRT backend</summary>

```python
model = OmniASR("omniasr_fp16.engine", "omniASR_tokenizer_written_v2.model", backend="tensorrt")
```

`backend` defaults to `"onnx"` and is never inferred from the file extension. The TensorRT backend binds reusable device buffers sized to the engine's profile maximum and runs inference with `set_tensor_address` + `execute_async_v3` on its own CUDA stream via `cuda-python` — buffers are allocated once and reused across calls; a length outside the engine's `[min, max]` profile raises `ValueError` before touching the GPU. `device`/`threads` are accepted but unused for this backend.
</details>

## Export and engine building

<details>
<summary>Requires a separate environment with PyTorch, fairseq2 and omnilingual-asr (not runtime dependencies)</summary>

```bash
# Export fine-tuned weights using their matching base architecture.
python export_onnx.py \
  --model omniASR_CTC_1B_v2 \
  --checkpoint checkpoints/step_100/model \
  --output artifacts/finetuned/model.onnx

python build_tensorrt.py artifacts/finetuned/model.onnx --output artifacts/omniasr_fp32.engine
python build_tensorrt.py artifacts/finetuned/model.onnx --precision fp16 --output artifacts/omniasr_fp16.engine
```

`--checkpoint` accepts a fairseq2-compatible custom model checkpoint file or the `model` directory within a native sharded fairseq2 step checkpoint. Incompatible checkpoint formats raise a fairseq2 model-checkpoint error. `--model` selects the base architecture and must match the fine-tuned weights. The checkpoint does not reliably carry enough architecture metadata to infer this safely.

Fine-tuned checkpoints are assumed to retain the base model's original tokenizer and token-ID mapping. The exporter infers and downloads that tokenizer from the model generation (`omniASR_CTC_*` uses v1; `omniASR_CTC_*_v2` uses written-v2), checks its vocabulary size against the CTC output as a compatibility sanity check, and saves it beside the graph as `tokenizer.model`. Equal vocabulary sizes do not establish tokenizer identity.

Fine-tuned export supports checkpoints whose architecture, CTC head, and vocabulary remain compatible with one of the eight supported OmniASR CTC model cards. Custom architectures, modified CTC heads or vocabularies, adapter-only checkpoints, LLM variants, and arbitrary Hugging Face model directories are not supported. Merge adapters into a full compatible checkpoint before exporting.

| Model/checkpoint | Support |
|---|---|
| Official 300M/1B/3B/7B, original and v2 | `from_pretrained()` |
| Compatible fine-tuned custom checkpoint file | `export_onnx.py --checkpoint` |
| Compatible native fairseq2 sharded model checkpoint | `export_onnx.py --checkpoint` |
| Modified architecture, CTC head, or vocabulary | Not supported |
| Adapter-only, LLM/seq2seq, or arbitrary HF model | Not supported |

Official exports remain available for reproducibility by omitting `--checkpoint`:

```bash
python export_onnx.py --model omniASR_CTC_300M_v2 --output artifacts/official/model.onnx
```

Supported base models are `omniASR_CTC_{300M,1B,3B,7B}` and `omniASR_CTC_{300M,1B,3B,7B}_v2`. Large ONNX exports may place tensor data in companion files; keep the entire output directory together when loading, publishing, or building an engine.

The custom-checkpoint path was integration-tested with [Peacockery/omni-ctc-300m-tajik](https://huggingface.co/Peacockery/omni-ctc-300m-tajik), loaded with the compatible `omniASR_CTC_300M_v2` configuration because its CTC head and tokenizer contain 10,288 entries. On a real speech clip, fairseq2 and ONNX Runtime produced the same frame count, greedy token IDs, and transcript; FP32 logits matched with `rtol=1e-4, atol=1e-3` (maximum absolute difference `0.000612`).

### Maintainer: publishing converted models

The runtime does not publish models. Maintainers can create a checksummed Hub bundle and publish it under the standard repository name with the repository tooling:

```bash
python tools/publish_hub.py \
  --model omniASR_CTC_1B_v2 \
  --onnx artifacts/1b-v2/model.onnx \
  --tokenizer omniASR_tokenizer_written_v2.model \
  --upload
```

Without `--upload`, this prepares and validates the bundle as a dry run. The publisher discovers ONNX external-data files, includes their sizes and SHA-256 digests in `config.json`, and uploads every required asset. `OmniASR.from_pretrained()` downloads and verifies all files declared by this manifest, so the same API works for large models once their repositories are published.

Tested with PyTorch 2.8.0, fairseq2 0.6, omnilingual-asr 0.2.0, ONNX 1.17.0, and TensorRT 10.16.1.11 (CUDA 12). The TensorRT profile is batch one, 16,000/80,000/480,000 MIN/OPT/MAX samples, TF32 disabled; engines are built for the local hardware/software stack and portability is not validated. `build_tensorrt.py` and `OmniASR.from_pretrained(..., backend="tensorrt")` call the same underlying `fast_omniasr.tensorrt_builder.build_engine`, so there's one build recipe, not two.

ONNX FP32 is the verified runtime; TensorRT FP32 and FP16 are explicit, separate experimental backends (not fallback-interchangeable). Mixed-precision TensorRT engines are ongoing stabilization work, not wired into `build_tensorrt.py` or the runtime yet.
</details>

## Performance and accuracy

TensorRT FP16 preserved **34/40** transcripts against FP32 on a small English/Turkish FLEURS sample; aggregate WER matched FP32 at 22.82% only because offsetting language-level errors hid the recognition changes. TensorRT FP32 preserved all 40 transcripts, though 3 clips failed a tight logit tolerance. None of this establishes production recognition parity, and 40 clips is too small a sample for general claims. The original experiment workspace holds the detailed evidence; it isn't bundled here.

## Benchmark and tests

```bash
python benchmarks/benchmark.py speech.wav --model dynamic.onnx --tokenizer tokenizer.model
python benchmarks/benchmark.py speech.wav --model omniasr_fp16.engine --tokenizer tokenizer.model --backend tensorrt

python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest tests/unit
```

Integration tests need real assets via env vars and skip otherwise: `OMNIASR_ONNX`/`OMNIASR_TOKENIZER`/`OMNIASR_TEST_WAV` for ONNX, `OMNIASR_TRT_FP32_ENGINE` (+ same tokenizer/WAV) and `OMNIASR_TRT_FP16_ENGINE`/`OMNIASR_TRT_FP16_TEST_WAV` for TensorRT. `tests/integration/test_hub.py` needs only `OMNIASR_TEST_WAV` but hits the real Hub repo over the network.

## License

Project code is [MIT](LICENSE) and does not bundle any model weights. Converted ONNX/tokenizer assets are published separately under Apache-2.0 for [300M](https://huggingface.co/EmreAkgul/omniASR-CTC-300M-ONNX), [300M v2](https://huggingface.co/EmreAkgul/omniASR-CTC-300M-v2-ONNX), [1B](https://huggingface.co/EmreAkgul/omniASR-CTC-1B-ONNX), [1B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-1B-v2-ONNX), [3B](https://huggingface.co/EmreAkgul/omniASR-CTC-3B-ONNX), [3B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-3B-v2-ONNX), [7B](https://huggingface.co/EmreAkgul/omniASR-CTC-7B-ONNX), and [7B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-7B-v2-ONNX), per the upstream [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) license (format conversion only, no retraining). Not affiliated with or endorsed by Meta or NVIDIA.
