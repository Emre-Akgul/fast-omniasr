# fast-omniasr

A standalone inference runtime for [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr), currently supporting exported `omniASR_CTC_300M_v2` CTC models with ONNX Runtime, without importing PyTorch, fairseq2 or Transformers. TensorRT engine building is available as an experimental conversion tool; the Python inference API currently supports ONNX only.

| Backend | 30 s latency | Sampled process VRAM | Exact transcripts vs fairseq2 FP32 |
|---|---:|---:|---:|
| fairseq2 FP16 | 137.6 ms | 1,822 MiB | 39/40 |
| TensorRT FP16 enabled | **79.8 ms** | **1,380 MiB** | 34/40 |
| TensorRT FP32 | 403.5 ms | 2,096 MiB | 40/40 |

Measured on an RTX 3060 Laptop GPU. **TensorRT FP16 is experimental and is not recognition-equivalent to FP32.** Latency is warmed inference on one continuous 30-second clip; transcript counts come from a separate 40-clip English/Turkish FLEURS sample. These are not public ONNX API timings. See [performance and accuracy](#performance-and-accuracy) for measurement scope and limitations.

## Installation

```bash
python -m pip install -e ".[onnx]"
```

For CUDA, install the `cuda` extra instead of `onnx`, provide compatible CUDA/cuDNN libraries, and use `device="cuda"`. Automatic CPU fallback is disabled.

## Usage

Supply a local exported ONNX model and its matching original SentencePiece tokenizer. Assets are not downloaded automatically by the runtime.

```python
from fast_omniasr import OmniASR

model = OmniASR("dynamic.onnx", "omniASR_tokenizer_written_v2.model")
print(model.transcribe("speech.wav").text)
```

For an existing NumPy waveform:

```python
result = model.transcribe_numpy(waveform, sample_rate=16000)
print(result.text)
```

Audio must be mono, 16 kHz, with at least 400 samples. The runtime applies whole-waveform normalization and greedy CTC decoding. Resampling, padded batching and language conditioning are not supported. Keep any ONNX external weight files alongside the model.

## Export and engine building

Conversion uses a separate environment with PyTorch, matching fairseq2/fairseq2n, omnilingual-asr and ONNX. The tested export stack was PyTorch 2.8.0, fairseq2 0.6, omnilingual-asr 0.2.0 and ONNX 1.17.0. These are not runtime dependencies.

```bash
python export_onnx.py --output artifacts/dynamic.onnx
```

The exporter loads the official model card and may download weights into fairseq2's cache. Obtain the matching `omniASR_tokenizer_written_v2.model` from the official assets separately.

TensorRT building requires a compatible TensorRT installation (tested with CUDA-12 TensorRT 10.16.1.11):

```bash
python build_tensorrt.py artifacts/dynamic.onnx --output artifacts/omniasr_fp32.engine
python build_tensorrt.py artifacts/dynamic.onnx --precision fp16 --output artifacts/omniasr_fp16.engine
```

The engine profile is batch one with MIN/OPT/MAX sample counts of 16,000/80,000/480,000. TF32 is disabled. FP16 must be explicitly requested. Engines are generated for the local hardware/software stack; portability is not validated.

## Performance and accuracy

Exploratory measurements on an RTX 3060 Laptop GPU processed a continuous 30-second clip in **79.8 ms with TensorRT FP16 enabled**, versus **137.6 ms with fairseq2 FP16**. Sampled TensorRT process VRAM was **1,380 MiB**. Those September 8, 2026 measurements include transfers and inference, but exclude audio preprocessing and decoding; they are not timings of the public ONNX API.

FP16 is experimental: TensorRT preserved **34/40** transcripts on a small English/Turkish FLEURS sample. Aggregate WER matched FP32 at 22.82%, but offsetting language-level errors hid recognition changes. TensorRT FP32 preserved all 40 transcripts; three clips failed tight logit tolerance. These findings do not establish production recognition parity. The original experiment workspace retains the detailed evidence; it is not bundled in this repository.

## Benchmark and tests

```bash
python benchmarks/benchmark.py speech.wav --model dynamic.onnx --tokenizer omniASR_tokenizer_written_v2.model
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest tests/unit
```

The benchmark measures end-to-end ONNX transcription, including file loading and decoding. Integration tests require `OMNIASR_ONNX`, `OMNIASR_TOKENIZER`, and `OMNIASR_TEST_WAV` pointing to the original five-second English `eng_cont_5s.wav` fixture. Without those assets they skip.

## License

Project code is licensed under [MIT](LICENSE). Upstream models, tokenizers and third-party software retain their own licenses and are not bundled. This is an independent project, not affiliated with Meta or NVIDIA.
