# Benchmark

```bash
python benchmarks/benchmark.py speech.wav --model dynamic.onnx --tokenizer omniASR_tokenizer_written_v2.model --runs 10
```

Use `--device cuda` for ONNX Runtime CUDA. This measures the public API end to end: file loading, normalization, inference and decoding. Model load, first call and warmed samples are reported separately.

These times are not directly comparable to the forward-only exploratory TensorRT measurements in the main README. The packaged benchmark supports ONNX; TensorRT inference benchmarking is not part of the public API yet.
