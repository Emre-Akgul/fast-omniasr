# fast-omniasr

A standalone inference runtime for [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr). Its CTC runtime runs all eight released CTC models (`300M`, `1B`, `3B`, and `7B`, both original and v2) on ONNX Runtime or TensorRT without PyTorch, fairseq2 or Transformers on the CTC inference path. A separate experimental LLM-ASR runtime uses PyTorch/TorchScript as a correctness baseline or ONNX Runtime for cached greedy decoding.

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

The requested provider/backend must initialize, and ONNX Runtime run-level fallback
is disabled. ORT may still assign supported graph plumbing, such as shape operations,
to its CPU EP. For a source checkout instead (e.g. to work on the library itself),
use `pip install -e ".[onnx,hub]"` from the repo root.

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

The first call downloads the ONNX asset, builds a local engine, and caches it under `~/.cache/fast-omniasr/tensorrt/<key>/model.engine`. The cache key includes the ONNX hash, precision, profile, TensorRT version, and GPU identity. The default profile accepts 400 to 480,000 samples and optimizes for 80,000 samples. Advanced users can pass `min_samples`, `opt_samples`, and `max_samples`; a changed profile creates a separate cached engine. `precision="fp16"` warns once per process about its accuracy caveat.

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
<summary>Requires a separate environment with PyTorch, fairseq2 and omnilingual-asr (not CTC runtime dependencies)</summary>

```bash
# Export fine-tuned weights using their matching base architecture.
python export_onnx.py \
  --model omniASR_CTC_1B_v2 \
  --checkpoint checkpoints/step_100/model \
  --output artifacts/finetuned/model.onnx

python build_tensorrt.py artifacts/finetuned/model.onnx --output artifacts/omniasr_fp32.engine
python build_tensorrt.py artifacts/finetuned/model.onnx --precision fp16 --output artifacts/omniasr_fp16.engine

# Optional profile tuning (defaults: 400/80,000/480,000 samples).
python build_tensorrt.py artifacts/finetuned/model.onnx \
  --min-samples 8000 --opt-samples 80000 --max-samples 480000 \
  --output artifacts/omniasr_custom.engine
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

Tested with PyTorch 2.8.0, fairseq2 0.6, omnilingual-asr 0.2.0, ONNX 1.17.0, and TensorRT 10.16.1.11 (CUDA 12). The current default TensorRT profile is batch one, 400/80,000/480,000 MIN/OPT/MAX samples with TF32 disabled; the performance figures above were measured with the earlier 16,000/80,000/480,000 profile. Engines are built for the local hardware/software stack and portability is not validated. `build_tensorrt.py` and `OmniASR.from_pretrained(..., backend="tensorrt")` call the same underlying `fast_omniasr.tensorrt_builder.build_engine`, so there's one build recipe, not two.

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

## Experimental LLM-ASR: 300M v2 greedy decoding

`fast_omniasr.llm.OmniASRLLM` is a separate runtime for
`omniASR_LLM_300M_v2`. It runs exported speech-encoder and decoder TorchScript graphs using PyTorch,
with no fairseq2 or omnilingual_asr runtime imports. The default full-prefix
baseline recomputes the prefix at every token; the opt-in cached decoder processes
the prefix once. The CTC backend interface is unchanged.

Install `pip install -e '.[llm]'` for inference. Export additionally requires a
working upstream `omnilingual_asr` / fairseq2 installation (tested with fairseq2
0.6 and PyTorch 2.8). Use a new artifact directory and a mono 16 kHz speech clip:

```bash
python export_llm.py --output /path/to/llm-300m-v2 --audio speech.wav
# Optional: --checkpoint /path/to/omniASR-LLM-300M-v2.pt
python tools/validate_llm.py /path/to/llm-300m-v2
```

The exporter checks the reference input syntax, variable audio/token lengths,
and generated token IDs against fairseq2 incremental greedy decoding. The report
records the maximum absolute logit error across every step, including EOS. It writes
`encoder.pt`, `decoder.pt`, `tokenizer.model`, `config.json`, `reference.json`,
and a byte-for-byte copy of the input clip with its lowercased source extension
(for example, `parity.wav` or `parity.flac`). The reference uses an artifact-relative
audio path, so the entire directory can be moved to another machine. Use
`tools/validate_llm.py /path/to/llm-300m-v2 --audio speech.wav` to override it;
the normalized audio hash must still match.
The separate validator loads those artifacts with reference-package imports
blocked, requires exact tokens and text, and rejects truncated generation.
Float32 weights require roughly 6.1 GiB of artifact storage in addition to the
source checkpoint; export and CPU inference need substantial RAM.

```python
from fast_omniasr.llm import OmniASRLLM

model = OmniASRLLM('/path/to/llm-300m-v2')
result = model.transcribe('speech.wav', max_new_tokens=512)
print(result.text)
print(result.stop_reason)  # "eos" means complete; limits are reported explicitly.
```

Scope: batch one, CPU float32 parity, clips up to 30 seconds, unspecified
language (the reference's learned language-zero embedding is retained).
An opt-in explicit KV-cache decoder is available alongside the full-prefix baseline.
No beam search, repetition/compression stopping, explicit language conditioning,
TensorRT, or Unlimited support yet. Greedy output is not expected
to match the upstream default five-beam output. Load only trusted model assets.

The export syntax follows the upstream
[model implementation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py)
and [generation implementation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/beamsearch.py).

Reference compatibility note: upstream 300M v2 configurations may expose a
9,812-entry `target_vocab_info` despite the checkpoint's 10,288 output classes.
The exporter validates the tokenizer against the output projection and preserves
the reference's language-marker index, recording both sizes in `config.json`.
It does not silently change the model's prompt syntax to repair this upstream
inconsistency.

Initial parity evidence is recorded in
[`benchmarks/llm_300m_v2_parity.json`](benchmarks/llm_300m_v2_parity.json):
a three-second LJSpeech excerpt matched all 47 text tokens through EOS against
fairseq2 incremental decoding on CPU float32. The first-step maximum absolute
logit difference was `3.8147e-6`; the maximum across all 48 decoder steps
(47 text tokens plus EOS) was `9.5367e-6`. Encoder checks additionally cover the 400-sample lower boundary, 0.5, 1, 2,
4, and 30 seconds, with varying decoder prefix lengths. This is a smoke test on one
speech excerpt, not a multilingual accuracy validation matrix.

Day 2 adds `decoder_cached.pt`, a TorchScript module with two methods:

```python
logits, cache = decoder.prefill(audio_context, bos)  # bos: int64 [1, 1]
logits, cache = decoder.decode_step(token, cache)    # token: int64 [1, 1]
```

`cache` is a list `[K0, V0, K1, V1, ...]`. Each tensor has shape
`[batch, cached_positions, kv_heads, head_dim]`; K contains interleaved RoPE,
V is the unrotated value projection. Cache length supplies the next absolute
position. Prefill processes the audio prefix and BOS once; each step projects only
one new token, attends to all cached positions, and returns new cache tensors.
The caller owns the cache; a new prefill starts an independent request.

```bash
python tools/export_cached_llm.py /path/to/llm-300m-v2 \
  --audio english.wav --audio turkish.wav
python tools/validate_llm.py /path/to/llm-300m-v2 --cached
```

```python
model = OmniASRLLM("/path/to/llm-300m-v2", cached=True)
result = model.transcribe("speech.wav")
```

The curated [Day 2 parity report](benchmarks/llm_300m_v2_cached_parity.json) records exact
three-way greedy equality on five clips (245 steps including EOS):

| Clip | Language | Steps including EOS |
| --- | --- | ---: |
| LJSpeech, 3 s | English | 48 |
| JFK, 5 s | English | 35 |
| Turkish sample, 5 s | Turkish | 73 |
| FLEURS excerpt, 3 s | English | 38 |
| FLEURS complete utterance, 4.2 s | Turkish | 51 |

The real decoder has 12 layers, 8 K/V heads, and head dimension 512. Across this
matrix the largest cached logit difference is `5.722e-6`, and the largest K/V
difference is `8.107e-6`. These are greedy decoding regression checks, not an
accuracy benchmark against human transcripts.

The default still loads `decoder.pt`. Both runtimes require only PyTorch and the
normal standalone dependencies. Exporting and checking the incremental oracle
requires fairseq2 and omnilingual_asr. See [cache validation and benchmarking](benchmarks/README.md#explicit-llm-kv-cache-cpu)
for the reproducible protocol and the distinction between generated fixtures and
curated benchmark records. On the four-thread i7-11800H CPU benchmark,
100 decoder steps fell from 229.92 s to 19.01 s (12.10×), including prefill and
excluding the encoder. Late cached steps stayed near 173–178 ms as output length
grew from 10 to 100; baseline late steps rose from 1,835 to 2,761 ms.
[Raw scaling measurements](benchmarks/llm_300m_v2_cache_scaling.json) include three
warm repetitions with identical forced tokens.

Day 3 exports the frozen Day 2 graphs to a PyTorch-free ONNX Runtime path:

```text
encoder.onnx:         audio [1, samples] -> context [1, audio_positions, 4096]
decoder_prefill.onnx: context + BOS -> logits + 24 K/V tensors
decoder_step.onnx:    one token + 24 K/V tensors -> logits + 24 updated K/V tensors
```

Each cache tensor is float32 `[1, cached_positions, 8, 512]`. Audio positions,
context length, and cached positions are dynamic. The decoder graphs refer to
one external `decoder.weights` file whose offsets and shapes are recorded in
`decoder_weights.json`; keep both files beside the graphs.

```bash
# ONNX-only inference
python -m pip install -e ".[onnx]"
python tools/validate_llm.py /path/to/day3 --backend onnx --device cpu
```

Use the `cuda` extra instead of `onnx` for CUDA-only inference.

Export and raw-tensor oracle validation additionally require Torch and the
`onnx` graph package:

```bash
python -m pip install -e ".[llm,onnx]" "onnx>=1.17,<2"
python export_llm_onnx.py /path/to/day2 /path/to/day3 --stage encoder
python export_llm_onnx.py /path/to/day2 /path/to/day3 --stage decoder
python tools/validate_llm_onnx.py /path/to/day2 /path/to/day3 --stage contexts
python tools/validate_llm_onnx.py /path/to/day2 /path/to/day3 --stage probes
python tools/validate_llm_onnx.py /path/to/day2 /path/to/day3 --stage generation
```

The exporter gates the encoder at 400 samples, 0.5, 1, 3, 5, and 30 seconds.
Decoder probes compare logits and every K/V tensor at cache lengths 10, 50,
150, and 300. Generation validation repeats those comparisons at every step of
the five English/Turkish Day 2 clips through EOS. Add `--device cuda` to each
validator command for the CUDAExecutionProvider. CUDA must initialize and ORT
run-level fallback is disabled; ORT may still place shape/control nodes on CPU.

```python
model = OmniASRLLM("/path/to/day3", backend="onnx", device="cpu")
result = model.transcribe("speech.wav")
```

The ONNX runtime imports NumPy, ONNX Runtime, and the tokenizer only. The
standalone validator actively blocks PyTorch, fairseq2, and omnilingual_asr
imports. On a 6 GiB GPU, it releases the encoder before loading decoder prefill,
then replaces prefill with the step session. Cache tensors remain explicit NumPy
values in this correctness-first implementation, so host/device cache copies and
cache concatenation remain optimization opportunities.

The [Day 3 parity record](benchmarks/llm_300m_v2_onnx_parity.json) records exact
CPU and CUDA greedy equality for all 245 generation steps, the raw cache error
bounds, blocked-import standalone runs, and the dynamic-length probes. See the
[ONNX LLM benchmark](benchmarks/README.md#onnx-llm-cache-cpu-and-cuda) for
separate encoder, prefill, per-token, 10/25/50/100-position, and memory results.

To rerun the real-artifact integration test:

```bash
FAST_OMNIASR_LLM_DIR=/path/to/llm-300m-v2 pytest tests/integration/test_llm.py -q
```

The integration test also accepts `FAST_OMNIASR_LLM_TEST_WAV` as an audio override.
CI keeps the lightweight CTC matrix and runs the LLM generation unit tests in a
separate Python 3.11 job with CPU PyTorch installed.

## License

Project code is [MIT](LICENSE) and does not bundle any model weights. Converted ONNX/tokenizer assets are published separately under Apache-2.0 for [300M](https://huggingface.co/EmreAkgul/omniASR-CTC-300M-ONNX), [300M v2](https://huggingface.co/EmreAkgul/omniASR-CTC-300M-v2-ONNX), [1B](https://huggingface.co/EmreAkgul/omniASR-CTC-1B-ONNX), [1B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-1B-v2-ONNX), [3B](https://huggingface.co/EmreAkgul/omniASR-CTC-3B-ONNX), [3B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-3B-v2-ONNX), [7B](https://huggingface.co/EmreAkgul/omniASR-CTC-7B-ONNX), and [7B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-7B-v2-ONNX), per the upstream [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) license (format conversion only, no retraining). Not affiliated with or endorsed by Meta or NVIDIA.
