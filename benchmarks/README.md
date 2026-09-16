# Benchmark

```bash
python benchmarks/benchmark.py speech.wav --model dynamic.onnx --tokenizer omniASR_tokenizer_written_v2.model --runs 10
```

Use `--device cuda` for ONNX Runtime CUDA. This measures the public API end to end: file loading, normalization, inference and decoding. Model load, first call and warmed samples are reported separately.

These times are not directly comparable to the forward-only exploratory TensorRT measurements in the main README. The packaged benchmark supports ONNX; TensorRT inference benchmarking is not part of the public API yet.

## Explicit LLM KV cache (CPU)

```bash
python tools/export_cached_llm.py /path/to/llm-300m-v2 \
  --audio english.wav --audio turkish.wav
python tools/validate_llm.py /path/to/llm-300m-v2 --cached
python tools/benchmark_llm_cache.py /path/to/llm-300m-v2 \
  --repeats 3 --output benchmarks/llm_300m_v2_cache_scaling.json
```

The exporter keeps `decoder.pt` and adds `decoder_cached.pt`. It first runs the
saved full-prefix graph in a separate process, then compares its greedy logits
with fairseq2 incremental decoding and the cached graph. By default, the real
fairseq2 decoder is reconstructed from the preserved baseline's exact weights;
`--checkpoint` optionally loads the original model instead. Every generated token,
including EOS, must agree; each layer's K/V tensors must also match within
`atol=2e-4, rtol=2e-4`. The original fixture must reproduce `reference.json`.
Additional clips are bundled with a digest in `cached_reference.json`.

Standalone validation blocks imports of fairseq2 and omnilingual_asr and validates
all bundled clips using the saved graph. The scaling benchmark uses identical
forced tokens at 10/25/50/100 steps, deliberately continuing past EOS. It excludes
the encoder and includes prefill, reports medians of repeated warm runs, and loads
the two decoder modes sequentially to bound memory. Each repetition runs 100
steps; the 10/25/50 measurements use prefixes of that same timed trajectory. It
reports total latency and the mean latency of the final five steps. Attention still reads the growing cache,
and this first implementation concatenates K/V tensors; cached decoding is not
constant-time or allocation-free.

Measured on an Intel Core i7-11800H with four CPU threads, float32, and three
warm repetitions ([raw measurements](llm_300m_v2_cache_scaling.json)):

| Output steps | Full-prefix seconds | Cached seconds | Speedup |
| ---: | ---: | ---: | ---: |
| 10 | 18.40 | 3.34 | 5.51× |
| 25 | 47.30 | 5.89 | 8.03× |
| 50 | 102.25 | 10.19 | 10.03× |
| 100 | 229.92 | 19.01 | 12.10× |

The final-five-step mean grows from 1,835 ms at 10 steps to 2,761 ms at 100
steps for the baseline. Cached steps remain near 173–178 ms. These decoder-only
measurements include prefill, exclude encoder/model loading, and use forced
identical tokens rather than stopping at the fixture's EOS. The shorter budgets
share samples with the 100-step runs, so they are not independent experiments.

The [five-clip parity report](llm_300m_v2_cached_parity.json) separately verifies
245 greedy steps including EOS against both regression oracles and records the
fresh-process standalone import-blocking check.

### Generated fixtures and curated records

`export_cached_llm.py` writes `cached_reference.json` inside the artifact directory.
That machine-generated fixture contains audio hashes, token IDs, EOS status,
cache dimensions/positions, and measured numerical errors. It is the input to
standalone validation, which emits one JSON object per clip on stdout:

```bash
python tools/validate_llm.py /path/to/llm-300m-v2 --cached > standalone.jsonl
```

`benchmarks/llm_300m_v2_cached_parity.json` is a **curated benchmark record**,
not the exporter's direct output. It combines the generated fixture with that
standalone output and annotations for language, source, validation date, test
status, and tolerances. Its `provenance` field lists these additions. Repeating
the protocol should reproduce the token comparisons; it does not promise a
byte-for-byte copy of the curated record. Hardware/date annotations in the
scaling record are also curated; the timing measurements come from the benchmark
tool.

The normal `llm-unit` CI job runs the Torch-only cache math tests for both MHA
and GQA. They use the production cache methods and an independent dense attention
oracle, block reference-package imports, and check saved/reloaded graphs at unseen
prefix lengths. Optional fairseq2 adapter coverage is retained separately:

```bash
pytest tests/integration/test_llm_cache_reference.py -q
```

## ONNX LLM cache (CPU and CUDA)

Day 3 keeps the Day 2 TorchScript files frozen and exports three ONNX graphs.
The decoder graphs share one external weight file; their K/V tensors remain
explicit runtime inputs and outputs.

```bash
# ONNX-only inference
python -m pip install -e ".[onnx]"
python tools/validate_llm.py /path/to/day3 --backend onnx --device cpu
```

Use the `cuda` extra instead of `onnx` for CUDA-only inference.

Export and raw-tensor oracle validation use the Torch development environment:

```bash
python -m pip install -e ".[llm,onnx]" "onnx>=1.17,<2"
python export_llm_onnx.py /path/to/day2 /path/to/day3 --stage encoder
python export_llm_onnx.py /path/to/day2 /path/to/day3 --stage decoder

python tools/validate_llm_onnx.py /path/to/day2 /path/to/day3 --stage contexts
python tools/validate_llm_onnx.py /path/to/day2 /path/to/day3 --stage probes
python tools/validate_llm_onnx.py /path/to/day2 /path/to/day3 --stage generation
```

Repeat all four validation commands with `--device cuda` using an
ONNX Runtime GPU installation. The contexts check covers the five bundled real
clips. The probe check uses cache lengths 10, 50, 150, and 300 and compares raw
logits plus all 24 cache outputs with the frozen `decoder_cached.pt`. The
generation check compares those tensors after every prefill/step call and
requires exact greedy tokens and EOS. The standalone check runs from the final
artifact while actively rejecting imports of PyTorch, fairseq2, and
omnilingual_asr. CUDA validation requires the CUDA EP to initialize and disables
ORT run-level fallback. ORT remains free to place graph-shape plumbing on CPU;
the validation does not claim that every node executes on CUDA.

Run the three backends in separate processes so peak resident memory remains
attributable to one runtime:

```bash
python tools/benchmark_llm_onnx.py /path/to/day2 /path/to/day3 \
  --backend torch --repeats 3 --output benchmarks/llm_300m_v2_onnx_torch_cpu.json
python tools/benchmark_llm_onnx.py /path/to/day2 /path/to/day3 \
  --backend onnx-cpu --repeats 3 --output benchmarks/llm_300m_v2_onnx_cpu.json
python tools/benchmark_llm_onnx.py /path/to/day2 /path/to/day3 \
  --backend onnx-cuda --repeats 3 --output benchmarks/llm_300m_v2_onnx_cuda.json
```

The benchmark warms each graph, excludes graph/session construction from phase
latencies, and reports that construction cost separately. Decoder totals include
prefill and use identical forced tokens at 10, 25, 50, and 100 output positions.
The shorter measurements are prefixes of each 100-position trajectory. Process
peak RSS, sampled CUDA process memory, and the theoretical float32 cache bytes at
100 positions are recorded. Cache concatenation and CPU-owned cache transfers are
part of the measured implementation.

Measured with four CPU threads on an Intel Core i7-11800H and, for CUDA, an RTX
3060 Laptop GPU with 6 GiB. Values are medians of three measured runs after one
warmup:

| Backend | 10 positions | 25 positions | 50 positions | 100 positions |
| --- | ---: | ---: | ---: | ---: |
| TorchScript CPU | 2.962 s | 5.461 s | 9.163 s | 16.978 s |
| ORT CPU | 2.995 s | 5.400 s | 9.519 s | 17.857 s |
| ORT CUDA | 0.586 s | 1.392 s | 2.835 s | 6.003 s |

| Backend | Encoder | Prefill | Mean step at 100 | Peak process RSS | Peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| TorchScript CPU | 567.4 ms | 1.560 s | 155.5 ms | 6,252 MiB | — |
| ORT CPU | 568.9 ms | 1.558 s | 164.8 ms | 6,170 MiB | — |
| ORT CUDA | 42.4 ms | 116.5 ms | 59.5 ms | 1,760 MiB | 5,330 MiB |

Each float32 K/V position is 384 KiB. The fixture has 151 audio positions, so
the cache is 94.125 MiB at 100 generated positions. CUDA arena shrinkage
is enabled after every dynamic-cache run; without it, ORT retained roughly 100
MiB per new cache shape and exhausted this 6 GiB device after five steps. Raw
measurements are in
[`llm_300m_v2_onnx_torch_cpu.json`](llm_300m_v2_onnx_torch_cpu.json),
[`llm_300m_v2_onnx_cpu.json`](llm_300m_v2_onnx_cpu.json), and
[`llm_300m_v2_onnx_cuda.json`](llm_300m_v2_onnx_cuda.json). The curated
[parity record](llm_300m_v2_onnx_parity.json) points to the generated raw-tensor
reports in the artifact directory.
