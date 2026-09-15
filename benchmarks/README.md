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
