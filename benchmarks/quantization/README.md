# INT8 and weight-only quantization of the CTC model (evaluation)

**Verdict.** Nothing quantized is published or supported yet; this is the evidence for
deciding whether to.

- **CPU: two variants pass every gate.** `int8-dynamic` is 1.9x faster than fp32 with 0.63x
  the memory and a quarter of the download, but its German WER sits 0.02 points under the
  limit (one more wrong word would fail it). `int8-weight-only` has more accuracy headroom
  and needs half the memory, but is 0.88x the speed of fp32.
- **GPU: none passes.** `trt-int8-mixed` keeps FP32-level accuracy but is slower than the
  shipped `trt-fp16` engine; `trt-int8` is 1.197x faster than it (gate: 1.2x) and loses
  accuracy on German and Central Kurdish. Quantized graphs on ONNX Runtime's CUDA provider
  are all slower than the published fp16 graph.
- **4-bit weights** fail the quality gate on 5 of 8 languages.

Shipping the CPU `int8-dynamic` graph would also take a published artifact, a runtime
`precision` option for it and its own FLEURS gate in CI; that is a maintainer decision,
not part of this evaluation.

`tools/quantize_ctc.py` builds the variants, scores them on the pinned FLEURS sample
(`benchmarks/fleurs/manifest.json`: 8 languages x 20 test utterances, 1,788 s of audio) and
applies the gates in [`gates.json`](gates.json). Model: `EmreAkgul/omniASR-CTC-300M-v2-ONNX`
at `6c1096d65763f1d268a7ed61fe8874596f0a4eb8` (the revision gated in
`benchmarks/fleurs/thresholds.json`).

```bash
python tools/quantize_ctc.py calibration --out calib/              # FLEURS dev clips
python tools/quantize_ctc.py quantize --ctc <export> --out variants/ \
    --calibration calib/calibration.json                           # ONNX variants (CPU)
python tools/quantize_ctc.py quantize --ctc <export> --out variants/ \
    --variants trt-fp16 trt-mixed trt-int8 trt-int8-mixed           # engines (CUDA, TRT 10)
python tools/quantize_ctc.py evaluate --ctc <export> --variants-dir variants/ \
    --device cpu --gate --report cpu.json
python tools/quantize_ctc.py evaluate --ctc <export> --variants-dir variants/ \
    --device cuda --gate --report cuda.json
```

`evaluate --gate` exits 1 unless some candidate passes every gate. Nothing here is
uploaded anywhere; the variants stay in the local `--out` directory.

## Variants

Only linear layers (a MatMul with a weight initializer, 146 in this model) are quantized.
Attention's activation-by-activation MatMuls stay float32 (quantizing them also made
static calibration run out of memory on a 16 GB machine), and so do the convolutional
front end and positional convolution, the layers
[`tensorrt_fp16_sensitivity.json`](../tensorrt_fp16_sensitivity.json) found to be
precision-sensitive.

| variant | what it is | runs on |
|---|---|---|
| `int8-dynamic` | ONNX Runtime dynamic quantization: int8 per-channel weights, activations quantized per call (`DynamicQuantizeLinear` + `MatMulInteger`) | CPU; on CUDA these ops have no kernel and run on the CPU |
| `int8-static` | int8 per-channel weights and int8 activations in QDQ form, symmetric, MinMax ranges calibrated on 16 FLEURS **dev** clips (2 per language, disjoint from the test sample) | CPU; CUDA EP dequantizes and runs float MatMuls |
| `int8-weight-only` | `MatMulNBits`, 8-bit symmetric weights in blocks of 32, float32 activations | CPU, CUDA |
| `int4-weight-only` | the same with 4-bit weights | CPU, CUDA |
| `trt-int8` | TensorRT 10 engine from the `int8-static` graph (explicit quantization), FP16 elsewhere | CUDA |
| `trt-int8-mixed` | as `trt-int8` with the front end pinned to FP32, like the shipped `mixed` recipe | CUDA |

Baselines measured in the same runs: the published `fp32` and `fp16` ONNX graphs and the
shipped TensorRT recipes `trt-fp16` and `trt-mixed` (built by `tensorrt_builder.build_engine`).

## Gates

A candidate ships only if it passes **both**:

1. **Quality**, per language, against the FP32 baselines in `thresholds.json` (which every
   FP32 run here reproduced exactly): WER may rise by at most max(1.0 point, 5 %) and CER
   by at most max(0.5 point, 5 %); Mandarin is gated on CER only. These margins are much
   tighter than the 3-point/15 % regression thresholds of `fleurs_eval.py`, because a
   quantized model is an opt-in trade-off whose cost must stay below the noise a user
   would notice, not just below "broken". With 20 utterances per language (357-837 reference
   words), 1 WER point is 4-8 words.
2. **Efficiency**, against the fastest variant already shipped on the same device (`fp32`
   ONNX on CPU, `trt-fp16` on CUDA): at least 1.2x faster over the whole set, or at most
   0.7x its peak memory (process RSS on CPU, process GPU memory on CUDA). A variant that is
   only smaller on disk, or slower than what users already have, isn't worth a second
   artifact to publish, test and support.

Latency is the median over 3 timed passes of the summed per-clip `transcribe` time
(preprocessing, inference, greedy decoding; batch size 1), after one warm-up clip. Each
variant runs in a fresh process; peak memory covers load, warm-up and all passes.

## Results: GPU (RTX 3060 Laptop, 6 GB; TensorRT 10.16, ONNX Runtime 1.23.2 CUDA EP)

The GPU was shared with other jobs during these runs, so the numbers were taken twice.
Transcripts were identical in both runs and the TensorRT timings agreed within 3 %;
ONNX Runtime CUDA timings varied more (up to 1.7x between passes in the first run). Shown
here: the second run ([cuda.json](cuda.json)); `int8-dynamic` comes from the first
([cuda-run1.json](cuda-run1.json)).

| variant | size | total s | RTF | GPU MiB | same text as fp32 | quality gate | efficiency vs `trt-fp16` |
|---|---:|---:|---:|---:|---:|---|---|
| `fp32` (ORT) | 1304 MB | 22.6 | 0.0127 | 3206 | 160/160 | reference | |
| `fp16` (ORT) | 678 MB | 15.2 | 0.0085 | 1164 | 159/160 | shipped | |
| `trt-fp16` | 658 MB | 4.8 | 0.0027 | 1306 | 139/160 | shipped | reference |
| `trt-mixed` | 683 MB | 6.7 | 0.0037 | 1428 | 148/160 | shipped | |
| **`trt-int8`** | 345 MB | **4.0** | 0.0022 | 1002 | 53/160 | **fail**: de_de, ckb_iq WER | 1.197x, 0.77 memory: **fail** (just) |
| **`trt-int8-mixed`** | 371 MB | 5.9 | 0.0033 | 1096 | 63/160 | **pass** | 0.82x, 0.84 memory: **fail** |
| `int8-weight-only` (ORT) | 404 MB | 24.9 | 0.0139 | 2180 | 143/160 | pass | 0.19x: fail |
| `int8-static` (ORT) | 367 MB | 25.8 | 0.0144 | 2180 | 63/160 | fail: de_de, fr_fr, cmn | 0.19x: fail |
| `int4-weight-only` (ORT) | 248 MB | 25.4 | 0.0142 | 2180 | 44/160 | fail: 5 languages | 0.19x: fail |
| `int8-dynamic` (ORT) | 367 MB | 120.7 | 0.0675 | 2180 | 109/160 | pass | 0.04x: fail |

ONNX Runtime GPU memory is its arena (it grew to the same 2180 MiB for every quantized
graph in this run, 1158 MiB for `int8-weight-only` in the first), so read it as an upper
bound. TensorRT figures are steady across runs.

Per-language WER / CER (Mandarin: CER only):

| variant | en_us | tr_tr | ckb_iq | de_de | es_419 | fr_fr | hi_in | cmn |
|---|---|---|---|---|---|---|---|---|
| fp32 | 20.88 / 5.43 | 22.01 / 4.08 | 42.30 / 8.71 | 12.99 / 3.29 | 9.29 / 2.69 | 18.51 / 5.75 | 12.66 / 5.57 | 21.84 |
| *gate (max)* | *21.92 / 5.93* | *23.11 / 4.58* | *44.41 / 9.21* | *13.99 / 3.79* | *10.29 / 3.19* | *19.51 / 6.25* | *13.66 / 6.07* | *22.93* |
| trt-fp16 | 21.81 / 5.70 | 22.01 / 4.08 | 42.58 / 8.81 | 12.99 / 3.29 | 9.29 / 2.69 | 18.74 / 5.79 | 12.66 / 5.57 | 22.08 |
| trt-mixed | 20.88 / 5.46 | 22.28 / 4.12 | 42.02 / 8.67 | 12.99 / 3.29 | 9.09 / 2.66 | 18.51 / 5.75 | 12.90 / 5.67 | 22.08 |
| trt-int8 | 20.88 / 5.54 | 21.74 / 4.16 | **44.82** / 9.13 | **14.46** / 3.47 | 9.49 / 2.73 | 19.19 / 5.68 | 13.26 / 5.88 | 22.79 |
| trt-int8-mixed | 20.19 / 5.58 | 22.28 / 4.05 | 42.30 / 8.81 | 12.99 / 3.21 | 9.68 / 2.73 | 18.74 / 5.72 | 12.78 / 5.62 | 22.08 |
| int8-weight-only | 21.11 / 5.54 | 22.28 / 4.12 | 42.58 / 8.76 | 12.99 / 3.29 | 8.89 / 2.66 | 18.74 / 5.72 | 12.66 / 5.62 | 22.08 |
| int8-static (ORT) | 20.88 / 5.74 | 22.55 / 4.23 | 41.74 / 8.85 | **14.46** / 3.59 | 9.68 / 2.69 | **19.86** / 6.13 | 11.83 / 5.27 | **23.03** |
| int4-weight-only | **22.97** / 5.85 | 22.83 / 4.16 | **45.38** / **9.36** | **14.95** / 3.70 | 9.29 / 2.86 | 19.19 / 5.83 | 13.38 / **6.23** | **23.63** |
| int8-dynamic (CUDA EP) | 20.88 / 5.46 | 21.74 / 4.12 | 43.70 / 8.85 | 13.24 / 3.29 | 9.68 / 2.69 | 18.06 / 5.75 | 12.90 / 5.67 | 22.08 |

What this shows:

- **TensorRT INT8 is the only GPU path with a real speedup**, and only a small one: the
  linear layers run in INT8, but at batch size 1 this model is not dominated by them.
  `trt-int8` takes 17 % less time than `trt-fp16` (1.197x, just under the 1.2x gate) and fails
  German (+1.47 WER) and Central Kurdish (+2.52 WER). Its losses follow the int8-static
  ONNX graph it was built from, so they come from the activation ranges, not TensorRT.
- **`trt-int8-mixed` passes the quality gate** (worst: +0.39 Spanish WER) but keeping the
  front end in FP32 costs more than INT8 saves: it is 0.82x the speed of `trt-fp16`. Against
  `trt-mixed`, its quality peer, it is 1.13x faster with 0.77x the GPU memory, which still
  misses both efficiency thresholds.
- **ONNX Runtime's CUDA provider gains nothing from quantization** here: `MatMulNBits`
  and QDQ graphs dequantize to float before the GEMM, so they are all slower than the
  published `fp16` graph, and dynamic quantization's `MatMulInteger` and
  `DynamicQuantizeLinear` have no CUDA kernels at all (the report's `cpu_placed_ops` lists
  146 + 98 of them on the CPU, hence 5x slower than fp32).
- **4-bit weights are too lossy** for this model on every device (fails 5 of 8 languages).

## Results: CPU

Intel Core i7-11800H laptop (8 cores / 16 threads), 4 ONNX Runtime threads (the package
default), ONNX Runtime 1.23.2. Shown: the second run ([cpu.json](cpu.json)), made with no
other job of this evaluation running and the variants in reverse order. In the first run
([cpu-run1.json](cpu-run1.json)) GPU work and other jobs shared the CPU and fp32's three
passes ranged from 250 to 416 s, so its speedups are inflated; the transcripts of every
variant were identical in both runs, and the same variants passed.

| variant | size | total s (3 passes) | RTF | peak RSS | same text as fp32 | quality gate | efficiency vs `fp32` |
|---|---:|---:|---:|---:|---:|---|---|
| `fp32` | 1304 MB | 260.4 (257-261) | 0.146 | 2460 MiB | 160/160 | reference | reference |
| **`int8-dynamic`** | 367 MB | 134.7 (118-140) | 0.075 | 1543 MiB | 99/160 | **pass** (de_de 13.97 vs 13.99) | **1.93x**, 0.63 memory: **pass** |
| **`int8-weight-only`** | 404 MB | 295.1 (287-308) | 0.165 | 1174 MiB | 143/160 | **pass** | 0.88x, **0.48 memory: pass** |
| `int8-static` | 367 MB | 196.6 (187-198) | 0.110 | 1631 MiB | 52/160 | fail: de_de WER 14.95 | 1.33x, 0.66 memory: pass |
| `int4-weight-only` | 248 MB | 254.3 (252-264) | 0.142 | 1050 MiB | 44/160 | fail: 5 languages | 1.02x, 0.43 memory: pass |

(`same text as fp32` comes from the first run, where fp32 ran first; transcripts did not
change between runs.)

Per-language WER / CER on CPU (gate row as in the GPU table):

| variant | en_us | tr_tr | ckb_iq | de_de | es_419 | fr_fr | hi_in | cmn |
|---|---|---|---|---|---|---|---|---|
| fp32 | 20.88 / 5.43 | 22.01 / 4.08 | 42.30 / 8.71 | 12.99 / 3.29 | 9.29 / 2.69 | 18.51 / 5.75 | 12.66 / 5.57 | 21.84 |
| *gate (max)* | *21.92 / 5.93* | *23.11 / 4.58* | *44.41 / 9.21* | *13.99 / 3.79* | *10.29 / 3.19* | *19.51 / 6.25* | *13.66 / 6.07* | *22.93* |
| int8-dynamic | 21.11 / 5.50 | 21.47 / 4.16 | 42.30 / 8.90 | 13.97 / 3.40 | 9.68 / 2.63 | 18.51 / 5.75 | 13.02 / 5.83 | 22.08 |
| int8-weight-only | 21.11 / 5.54 | 22.28 / 4.12 | 42.58 / 8.76 | 12.99 / 3.29 | 8.89 / 2.66 | 18.74 / 5.72 | 12.66 / 5.62 | 22.08 |
| int8-static | 19.95 / 5.54 | 22.01 / 4.16 | 42.30 / 8.99 | **14.95** / 3.47 | 9.88 / 2.86 | 19.41 / 6.02 | 12.66 / 5.42 | 22.55 |
| int4-weight-only | **22.97** / 5.85 | 22.83 / 4.16 | **45.38** / **9.36** | **14.95** / 3.70 | 9.29 / 2.86 | 19.19 / 5.83 | 13.38 / **6.23** | **23.63** |

What this shows:

- **Dynamic INT8 is the clear CPU win**: the 146 linear layers run as `MatMulInteger`,
  which is about twice as fast as float32 on this AVX-512 VNNI CPU, and activations are
  quantized per call, so no calibration data is needed. Its accuracy cost is small but
  uneven: German loses 0.98 WER points (4 words more out of 408), right at the gate.
- **Static INT8** is both slower than dynamic in this ONNX Runtime version (not
  investigated further) and less accurate: with activation ranges fixed from 16
  calibration clips, German loses 1.96 WER points.
- **Weight-only INT8** saves memory but not time on CPU: with the default accuracy level,
  `MatMulNBits` dequantizes the weights to float32 for each call and computes in float32.
- Timings on this laptop drift with temperature and background load by up to ~10 % between
  passes; the speedups above are well clear of that, except the weight-only ones.

## What would change the verdict

- Batch inference: INT8 GEMMs pay off with larger M; with the batch graphs from
  `export_onnx.py --batch` the TensorRT INT8 speedup may exceed the gate. Not measured here.
- Better INT8 activation ranges (percentile or entropy calibration, more calibration
  clips, per-layer exclusions found by sensitivity analysis like the FP16 one) might bring
  `trt-int8` within the quality gate without the FP32 front end.
- A GPU with faster INT8 tensor cores relative to FP16 (Ada/Hopper with FP8/INT8) could
  move the efficiency numbers; the gates, not these numbers, are the policy.

Re-run the tool on the target hardware before publishing anything; any variant that is
shipped later needs its own `thresholds.json` entry and CI gate like the FP32 model.
