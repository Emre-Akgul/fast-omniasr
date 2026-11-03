# LLM-ASR export history: the 300M v2 baseline

> **Historical developer notes.** This is the development log of the LLM-ASR export
> (the TorchScript baseline, then the explicit KV cache on "Day 2" and the first ONNX
> path on "Day 3"). It records how parity was established and the reports produced at
> each stage. It is **not** a description of current behaviour: the current runtime, its
> defaults and its performance are documented in the [README](../README.md#llm-asr-models).
> Notes marked *Since then* point out what has changed.

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

Scope of the TorchScript baseline: batch one, CPU float32, clips up to 30 seconds.
An opt-in explicit KV-cache decoder is available alongside the full-prefix baseline.
No beam search, repetition/compression stopping, TensorRT, or Unlimited support yet.

> *Since then:* all eight standard LLM models (300M–7B, original and v2) are exported and
> published; see the README for the current support matrix.

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
[`benchmarks/llm_300m_v2_parity.json`](../benchmarks/llm_300m_v2_parity.json):
a three-second LJSpeech excerpt matched all 47 text tokens through EOS against
fairseq2 incremental decoding on CPU float32. The first-step maximum absolute
logit difference was `3.8147e-6`; the maximum across all 48 decoder steps
(47 text tokens plus EOS) was `9.5367e-6`. Encoder checks additionally cover the 400-sample lower boundary, 0.5, 1, 2,
4, and 30 seconds, with varying decoder prefix lengths. This is a smoke test on one
speech excerpt, not a multilingual accuracy validation matrix.

## Day 2: explicit KV cache

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

The curated [Day 2 parity report](../benchmarks/llm_300m_v2_cached_parity.json) records exact
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
requires fairseq2 and omnilingual_asr. See [cache validation and benchmarking](../benchmarks/README.md#explicit-llm-kv-cache-cpu)
for the reproducible protocol and the distinction between generated fixtures and
curated benchmark records. On the four-thread laptop CPU benchmark,
100 decoder steps fell from 229.92 s to 19.01 s (12.10×), including prefill and
excluding the encoder. Late cached steps stayed near 173–178 ms as output length
grew from 10 to 100; baseline late steps rose from 1,835 to 2,761 ms.
[Raw scaling measurements](../benchmarks/llm_300m_v2_cache_scaling.json) include three
warm repetitions with identical forced tokens.

## Day 3: ONNX Runtime

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

> *Since then:* the ONNX runtime uses a single unified `decoder.onnx` for prefill and every
> step, keeps sessions loaded across calls, and on CUDA keeps the K/V cache on the GPU
> through I/O binding. The one-session-at-a-time behaviour described above is now opt-in
> (`low_memory=True`). Published repos are loaded with `OmniASRLLM.from_pretrained()`.

The [Day 3 parity record](../benchmarks/llm_300m_v2_onnx_parity.json) records exact
CPU and CUDA greedy equality for all 245 generation steps, the raw cache error
bounds, blocked-import standalone runs, and the dynamic-length probes. See the
[ONNX LLM benchmark](../benchmarks/README.md#onnx-llm-cache-cpu-and-cuda) for
separate encoder, prefill, per-token, 10/25/50/100-position, and memory results.

To rerun the real-artifact integration test:

```bash
FAST_OMNIASR_LLM_DIR=/path/to/llm-300m-v2 pytest tests/integration/test_llm.py -q
```

The integration test also accepts `FAST_OMNIASR_LLM_TEST_WAV` as an audio override.
CI keeps the lightweight CTC matrix and runs the LLM generation unit tests in a
separate Python 3.11 job with CPU PyTorch installed.
