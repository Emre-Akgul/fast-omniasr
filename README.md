# fast-omniasr

A standalone inference runtime for [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr). Runs exported `omniASR_CTC_300M_v2` models on ONNX Runtime or TensorRT — no PyTorch, fairseq2 or Transformers on the inference path.

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
python -m pip install -e ".[onnx]"      # ONNX Runtime CPU
python -m pip install -e ".[cuda]"      # ONNX Runtime GPU (device="cuda")
python -m pip install -e ".[tensorrt]"  # cuda-python; also requires a separate TensorRT install
```

Automatic CPU fallback is disabled — the requested provider/backend must actually initialize.

## Usage

```python
from fast_omniasr import OmniASR

model = OmniASR("dynamic.onnx", "omniASR_tokenizer_written_v2.model")
print(model.transcribe("speech.wav").text)

# or from a NumPy waveform
result = model.transcribe_numpy(waveform, sample_rate=16000)
```

Audio must be mono, 16 kHz, at least 400 samples. Assets (model + tokenizer) are supplied locally, never downloaded automatically.

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
python export_onnx.py --output artifacts/dynamic.onnx
python build_tensorrt.py artifacts/dynamic.onnx --output artifacts/omniasr_fp32.engine
python build_tensorrt.py artifacts/dynamic.onnx --precision fp16 --output artifacts/omniasr_fp16.engine
```

Tested with PyTorch 2.8.0, fairseq2 0.6, omnilingual-asr 0.2.0, ONNX 1.17.0, and TensorRT 10.16.1.11 (CUDA 12). Obtain `omniASR_tokenizer_written_v2.model` from the official assets separately. The TensorRT profile is batch one, 16,000/80,000/480,000 MIN/OPT/MAX samples, TF32 disabled; engines are built for the local hardware/software stack and portability is not validated.

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

Integration tests need real assets via env vars and skip otherwise: `OMNIASR_ONNX`/`OMNIASR_TOKENIZER`/`OMNIASR_TEST_WAV` for ONNX, `OMNIASR_TRT_FP32_ENGINE` (+ same tokenizer/WAV) and `OMNIASR_TRT_FP16_ENGINE`/`OMNIASR_TRT_FP16_TEST_WAV` for TensorRT.

## License

Project code is [MIT](LICENSE). Upstream models, tokenizers and third-party software retain their own licenses and are not bundled. Independent project, not affiliated with Meta or NVIDIA.
