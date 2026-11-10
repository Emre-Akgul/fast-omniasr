# fast-omniasr

A standalone inference runtime for [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr). Its CTC runtime runs all eight released CTC models (`300M`, `1B`, `3B`, and `7B`, both original and v2) on ONNX Runtime or TensorRT without PyTorch, fairseq2 or Transformers on the CTC inference path. A separate LLM-ASR runtime runs the eight standard LLM models (`300M`, `1B`, `3B`, and `7B`, original and v2) with KV-cached greedy decoding on ONNX Runtime, loaded with `OmniASRLLM.from_pretrained()`; PyTorch/TorchScript graphs remain available as a correctness baseline.

## Benchmark

30-second clip on a consumer laptop GPU, through the public `OmniASR` API (`benchmarks/benchmark.py`, warm median of 10 runs, includes file load, normalization, inference and decoding):

| Backend | Latency | Exact transcripts vs FP32 (40-clip FLEURS sample) |
|---|---:|---:|
| ONNX Runtime CUDA (FP32) | 358 ms | reference |
| TensorRT FP32 | 374 ms | 40/40 |
| TensorRT FP16 | **76 ms** | 34/40¹ |

¹ An earlier TensorRT and profile; on the current stack FP16 keeps 141/160 transcripts identical to FP32 on the larger FLEURS sample below.

**TensorRT FP16 trades a few near-tie token flips for speed.** On the 160-utterance, 8-language FLEURS sample with the current TensorRT, FP16 keeps 141/160 transcripts identical to FP32 and every language's WER within 0.5 points; `precision="mixed"` keeps the precision-sensitive convolutional front end in FP32 (150/160 identical, about twice the FP16 latency, still ~2.6× faster than FP32). The latency table above predates these measurements. See [Performance and accuracy](#performance-and-accuracy).

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
to its CPU EP. On CUDA, cuDNN uses its fixed convolution algorithm, because newer cuDNN
(9.27+) fails to allocate for inputs under ~2.6 s when algorithms are searched (#97). ONNX
Runtime logs `OP Conv(...) running in Fallback mode. May be extremely slow.` for each new
input length as a result. It is expected and costs ~5% on long clips; hide it with
`onnxruntime.set_default_logger_severity(3)`. For a source checkout instead (e.g. to work on the library itself),
use `pip install -e ".[onnx,hub]"` from the repo root.

## Usage

```python
from fast_omniasr import OmniASR

model = OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="onnx")
print(model.transcribe("speech.wav").text)
```

`from_pretrained` downloads `model.onnx`/`tokenizer.model` via `huggingface_hub` (requires the `hub` extra), verifies each against the sha256 in the repo's `config.json`, and stores them as regular files under `~/.cache/fast-omniasr/hub` (or `cache_dir`) — later calls (even fully offline, e.g. `HF_HUB_OFFLINE=1`) reuse them. Regular files are required because ONNX Runtime 1.31+ rejects external tensor data reached through the Hugging Face cache's symlinks. A branch or tag is resolved to one commit for the whole download. Each repo revision's bundle is assembled in a staging directory and swapped in under a cross-process lock, so concurrent loaders of the same bundle download it once and an interrupted or failed download leaves the previously published bundle intact. Files are opened after `from_pretrained` returns, so a process that loads while another publishes a newer commit can still briefly race that swap.

Offline use and the cache:

```python
from fast_omniasr import OmniASR, hub

model = OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", local_files_only=True)
print(model.snapshot)  # Snapshot(repo_id=..., revision='main', commit='6c1096d…', path=...)

for bundle in hub.list_cached():  # every downloaded repo revision, with commit and size
    print(bundle.repo_id, bundle.revision, bundle.commit, bundle.size_bytes)
hub.delete_cached("EmreAkgul/omniASR-CTC-300M-v2-ONNX")  # "main"; or a revision, or all_revisions=True
```

`local_files_only=True` (on `OmniASR.from_pretrained` and `OmniASRLLM.from_pretrained`) never contacts the Hub: it verifies and loads the cached copy of the revision (for a branch or tag, the commit recorded when it was downloaded) or raises `hub.LocalBundleNotFoundError` saying which files are missing (e.g. another precision was downloaded) or fail verification. It also works on a read-only cache. `model.snapshot` records the repo, the requested revision, the commit it resolved to and the local directory. `delete_cached` takes the same lock as downloads, so it never removes a bundle mid-download. Download failures leave the cache unchanged; a checksum mismatch names the `delete_cached` call that clears a damaged copy. These helpers live in `fast_omniasr.hub`.

From the command line, after installing the `onnx` (or `cuda`) and `hub` extras (`pip install "fast-omniasr[onnx,hub]"`):

```bash
fast-omniasr transcribe speech.wav --model EmreAkgul/omniASR-CTC-300M-v2-ONNX
fast-omniasr transcribe meeting.wav --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --vad --json
fast-omniasr transcribe clip.wav --model EmreAkgul/omniASR-LLM-300M-v2-ONNX --kind llm --lang eng_Latn
fast-omniasr languages --model EmreAkgul/omniASR-LLM-300M-v2-ONNX
fast-omniasr inspect --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --json
fast-omniasr diagnose
```

Output is UTF-8. With `--json`, `transcribe` writes one JSON object per line, one per audio file, with the same fields as the result object plus `path`. A file that fails gets `path` and `error` instead, the rest still run, and the exit code is 1. `inspect` and `diagnose` write one JSON object and `languages` one JSON array. Options for the other model kind, such as `--lang` on CTC, are usage errors. `--chunk-length` and `--vad` imply `--long`, which uses `transcribe_long`. `diagnose` loads no model and lists versions, ONNX Runtime providers, whether TensorRT imports, and cached models. `inspect` loads the model, so it needs the same extras as `transcribe`. `languages` downloads only the code table and prints codes in lower case, and `--lang` accepts either case. `python -m fast_omniasr` works the same way.

### Precision (fp32, fp16, bf16)

Every published model — CTC and LLM — ships float32 graphs plus `fp16/` and `bf16/`
variants. Pass `precision=` to `from_pretrained`; only that variant (and the shared
tokenizer) is downloaded:

```python
model = OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-1B-v2-ONNX", device="cuda", precision="fp16")
```

| `precision` | Devices | What changes |
|---|---|---|
| `fp32` (default) | CPU, CUDA | Reference; matches fairseq2 |
| `fp16` | CPU, CUDA | MatMul weights in half precision: roughly half the download and weight memory, ~1.5-2.5x faster on GPU; no speedup on CPU |
| `bf16` | CUDA on Ampere or newer GPUs | Same as fp16 with bfloat16; ONNX Runtime has no CPU bf16 kernels |

Only the MatMul weights are converted (`tools/convert_precision.py`): convolutions,
normalization, softmax, the LLM K/V cache and all inputs/outputs stay float32, because a
fully fp16 graph overflowed in the LLM decoder. On English, Turkish and Central Kurdish test
clips, fp16 transcripts matched fp32 for every model; bf16 changed one word in a few runs
(e.g. "count" -> "county"), the kind of drift upstream's own bf16 default shows. Each repo's
model card has its numbers and `parity/precision_agreement.json`.

| On a data-center GPU (warm, per clip) | fp32 | fp16 |
|---|---:|---:|
| CTC 1B v2 | 5.4 GiB, ~90 ms | 3.4 GiB, ~48 ms |
| CTC 7B v2 | 26.5 GiB, ~368 ms | 13.4 GiB, ~141 ms |
| LLM 300M v2 | 9.5 GiB | 6.5 GiB, ~1.5x faster |
| LLM 7B v2 | 33.5 GiB | 19.5 GiB, ~1.5x faster |

For the fast path with no manual export or `build_tensorrt.py` call, pass `backend="tensorrt"` and an explicit `precision` — `"fp32"`, `"fp16"` or `"mixed"` (no "auto" — see [Performance and accuracy](#performance-and-accuracy) before choosing a reduced precision):

```python
model = OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="tensorrt", precision="fp16")
```

The first call downloads the ONNX asset, builds a local engine, and caches it under `~/.cache/fast-omniasr/tensorrt/<key>/model.engine`. The cache key includes the SHA-256 of the ONNX graph and of every external-data weights file it references, the precision, profile, TensorRT version and the CUDA major it was built for (`cu12`/`cu13`), and the identity (name and compute capability) of the CUDA device it is built on (`device_index`, or the current device), so replacing only a weights file, switching GPU model, or switching TensorRT's CUDA flavor rebuilds the engine. The driver version, device index and `cuda-python` version are not part of the key. Each cached engine's size and SHA-256 are recorded and checked before reuse: a corrupted or replaced engine is rebuilt with a warning, and an engine TensorRT cannot deserialize is rebuilt once before `from_pretrained` gives up. The default profile accepts 400 to 480,000 samples and optimizes for 80,000 samples. Advanced users can pass `min_samples`, `opt_samples`, and `max_samples`; a changed profile creates a separate cached engine. `precision="fp16"` warns once per process about its accuracy caveat; `"mixed"` builds a separate engine with the feature extractor and positional convolution kept in FP32. TensorRT 11 removed the builder's precision flags and only builds strongly typed networks. There, `fp16` engines are built from a float16 copy of the graph made with ONNX Runtime's float16 converter, which needs the `onnx` package and ONNX Runtime (`pip install onnx "fast-omniasr[onnx]"`, or `[cuda]`). The conversion holds the fp32 graph and its fp16 copy in memory, about twice the size of the fp32 graph (roughly 2.5 GB for 300M and 55 GB for 7B), and writes the fp16 copy beside the cached engine until the build finishes. `mixed` is refused on TensorRT 11 before anything is downloaded, because a float16 graph with the front end kept in FP32 came out about as far from FP32 as plain fp16 (#101). TensorRT 10 keeps the flag-based recipe for all three precisions.

To tune the build, pass `build_options` (or the matching `build_tensorrt.py` flags):

```python
from fast_omniasr import OmniASR, TensorRTBuildOptions

options = TensorRTBuildOptions(
    workspace_bytes=2 * 1024**3,          # workspace memory limit per layer (default 1 GiB)
    optimization_level=4,                 # 0 to 5 (default: TensorRT's own, currently 3)
    timing_cache="trt-timing.cache",      # reused and updated by each build
)
model = OmniASR.from_pretrained(repo, backend="tensorrt", precision="fp16", build_options=options)
```

A timing cache roughly halves rebuild time or better. It is only an optimization: a missing, corrupt or unwritable file costs a warning at most, and the build still finishes. Builds sharing one file replace it whole, so the last to finish wins. Non-default options are part of the cache key, so each combination gets its own engine; using a timing cache counts, but not its contents or path. With the defaults, existing cached engines stay valid.

`OmniASR` and `OmniASRLLM` release their ONNX Runtime sessions or TensorRT engine on `close()` or when leaving a `with` block (`with OmniASR.from_pretrained(...) as model:`); `close()` is idempotent, and transcribing afterwards raises `RuntimeError`. An `OmniASR` instance may be shared across threads. With the TensorRT backend, each instance owns one execution context and its buffers, so concurrent calls are serialized (never mixed); create one instance per worker for parallel GPU throughput. `OmniASRLLM` instances may also be shared: each `transcribe()` runs its encode and full decode loop alone (calls are serialized), so concurrent requests never mix; use one instance per worker for parallel throughput.

[`examples/server`](examples/server) is a tested Docker deployment: a dependency-free HTTP service with model preload, health and readiness endpoints, input limits, a bounded request queue and graceful shutdown. In a long-running server, load the model once and keep it. On the discrete GPUs we measured, GPU memory stayed flat across hundreds of requests on the ONNX Runtime CUDA path (CTC, and the LLM both resident and with `low_memory=True`). ONNX Runtime's default arena reserves one large region on the first request and reuses it. Adding the per-run `memory.enable_memory_arena_shrinkage` run option makes no difference there, because there is nothing above that region to free. It only frees memory when the arena grows in exact-size blocks (`arena_extend_strategy=kSameAsRequested`), and then it frees and re-allocates on every request. On integrated GPUs that share memory with the host, that pattern has been reported to leave driver memory behind on every request (#96). Check your own device and settings with:

```bash
python tools/gpu_memory_soak.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --device cuda               # same request 60 times
python tools/gpu_memory_soak.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --device cuda \
    --arena-extend-strategy kSameAsRequested --arena-shrinkage                                      # free and re-allocate per request
python tools/gpu_memory_soak.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --device cuda --mixed --seconds 30
```

After every request it samples NVML process and device memory, `NvMapMemUsed` from `/proc/meminfo` (where integrated GPUs' allocations show up), this process's cgroup `kernel` memory and RSS. Exit status: 1 if any counter, RSS included, keeps growing after warm-up, 2 for invalid options, 3 if the model fails to load or run, 4 if no GPU counter is available (RSS can't see GPU memory). `--gpu-mem-limit` caps the ONNX Runtime arena to test a bounded configuration.

Or supply local assets directly (no download, no `hub` extra needed):

```python
model = OmniASR("dynamic.onnx", "omniASR_tokenizer_written_v2.model")
result = model.transcribe_numpy(waveform, sample_rate=16000)  # or from a NumPy waveform
```

Audio must be mono, 16 kHz, at least 400 samples; anything else is rejected, so what's transcribed is exactly what you passed in. To accept other formats, opt in with `convert_audio=True` on `transcribe`, `transcribe_numpy` or `OmniASRLLM.transcribe`:

```python
# pip install "fast-omniasr[resample]"
result = model.transcribe("call-44khz-stereo.wav", convert_audio=True)
result = model.transcribe_numpy(waveform, sample_rate=48000, convert_audio=True)
```

This averages the channels of `(samples, channels)` audio and resamples to 16 kHz with [soxr](https://github.com/dofuuz/python-soxr). The channel order of NumPy input isn't guessed: an array with more channels than samples is rejected. The length limits apply to the converted audio.

For recordings longer than about 15 seconds, use `transcribe_long`, on `OmniASR` and `OmniASRLLM`. It cuts the audio every `chunk_length` seconds (default 15), transcribes the chunks and joins the text in order. There's no overlap handling, so a word on a cut can be damaged:

```python
result = model.transcribe_long("meeting.wav")
print(result.text)
for chunk in result.chunks:
    print(chunk.start, chunk.end, chunk.text)
```

Pass `vad_filter=True` to cut at silence and skip non-speech, using Silero VAD as faster-whisper does (needs the `onnx` extra). `chunks` keep their times in the original recording. When the VAD finds no speech, the result has empty text, though non-speech it mistakes for speech, such as some tones, is still transcribed. It lowered CTC error on the test recordings, which have clean pauses, though not on every language.

The default is a safe choice rather than a proven best, and it wasn't measured on continuous speech from one speaker. On test recordings that change speaker every few seconds, CTC on chunks past about 20 seconds lost whole spans of speech on some languages, for reasons not yet known. See [docs/long-audio.md](docs/long-audio.md).

For CTC models, pass `word_timestamps=True` to `transcribe`, `transcribe_numpy` or `transcribe_long` to get word and token times, as faster-whisper does. In `transcribe_long` they are times in the original recording, even with `vad_filter=True`:

```python
result = model.transcribe("speech.wav", word_timestamps=True)
for word in result.words:
    print(word.start, word.end, word.word)
```

They have 20 ms resolution, and a word's times cover only the frames where the model emitted it, not its full sound. See [docs/timestamps.md](docs/timestamps.md).

The LLM's output is the model's own, repeats included: nothing is detected or filtered. On audio with no speech it can loop until `max_new_tokens`, which `stop_reason` reports. To discourage that, pass `repetition_penalty`, which works as in faster-whisper: `model.transcribe(audio, lang="eng_Latn", repetition_penalty=1.1)`. Values above 1 penalize tokens already generated, and the default of 1 changes nothing.

Use it only where loops are the bigger problem, because it can damage real transcripts. On the test clips, 1.1 and 1.2 ended the loops on silence and hum and left a single English sentence unchanged, even said four times, but they changed the text of a two-language clip. EOS is never penalized, so wherever a repeated token only narrowly beats it, the transcript ends there: on a 30-second clip, both values dropped the second sentence, about 40% of the output. It applies to the LLM only: CTC has no generation loop.

The tokenizer must be the one the graph was exported with. On load, the model's output vocabulary has to match the tokenizer's size. Graphs exported by `export_onnx.py` also record the tokenizer's sha256 in their ONNX metadata, so a different tokenizer of the same size is rejected too. Older exports can be stamped in place with `python tools/stamp_tokenizer.py model.onnx fp16/model.onnx --tokenizer tokenizer.model`; fp16/bf16 variants made by `tools/convert_precision.py` keep the stamp. Stamping changes the graph file, so a published repo needs its `config.json` checksums regenerated (republish it). The check compares the exact tokenizer file bytes. A TensorRT engine carries no metadata, so a local engine gets the vocabulary-size check only; `from_pretrained(backend="tensorrt")` checks the ONNX graph before building the engine.

<details>
<summary>TensorRT backend</summary>

```python
model = OmniASR("omniasr_fp16.engine", "omniASR_tokenizer_written_v2.model", backend="tensorrt")
```

`backend` defaults to `"onnx"` and is never inferred from the file extension. The TensorRT backend binds reusable device buffers sized to the engine's profile maximum and runs inference with `set_tensor_address` + `execute_async_v3` on its own CUDA stream via `cuda-python` — buffers are allocated once and reused across calls; a length outside the engine's `[min, max]` profile raises `ValueError` before touching the GPU. It runs on the CUDA device given by `device_index` (default: the current device), so `device` must be omitted or `"cuda"` and `threads` must be omitted; options a backend would ignore raise `ValueError` instead. `device` (`"cpu"`/`"cuda"`) and `threads` (a positive integer, default 4) are validated the same way for the ONNX CTC and LLM runtimes; the TorchScript LLM baseline rejects `threads` and `low_memory`. `device_index` picks the GPU, as in faster-whisper: `OmniASR(..., device="cuda", device_index=1)`, `OmniASR.from_pretrained(..., device="cuda", device_index=1)` (or `backend="tensorrt"`) and `OmniASRLLM.from_pretrained(..., device="cuda", device_index=1)`, or `--device-index` on the command line. It sets ONNX Runtime's `device_id`, keeps the LLM's GPU-resident K/V cache on that GPU, and builds, caches and loads TensorRT engines on it. When TensorRT switches devices it returns the calling thread to its previous device only if that device was already in use, so a process pinned to GPU 1 doesn't also open a context on GPU 0; a thread that had not chosen a GPU is left on `device_index`. The default (`None`) is GPU 0 for ONNX Runtime and the current CUDA device for TensorRT, as before. It needs `device="cuda"`, and an index with no GPU behind it raises an error.
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
python build_tensorrt.py artifacts/finetuned/model.onnx --precision mixed --output artifacts/omniasr_mixed.engine

# Optional profile tuning (defaults: 400/80,000/480,000 samples).
python build_tensorrt.py artifacts/finetuned/model.onnx \
  --min-samples 8000 --opt-samples 80000 --max-samples 480000 \
  --output artifacts/omniasr_custom.engine
```

`--checkpoint` accepts a fairseq2-compatible custom model checkpoint file or the `model` directory within a native sharded fairseq2 step checkpoint. Incompatible checkpoint formats raise a fairseq2 model-checkpoint error. `--model` selects the base architecture and must match the fine-tuned weights. The checkpoint does not reliably carry enough architecture metadata to infer this safely.

Fine-tuned checkpoints are assumed to retain the base model's original tokenizer and token-ID mapping. The exporter infers and downloads that tokenizer from the model generation (`omniASR_CTC_*` uses v1; `omniASR_CTC_*_v2` uses written-v2), checks its vocabulary size against the CTC output as a compatibility sanity check, and saves it beside the graph as `tokenizer.model`. Equal vocabulary sizes do not establish tokenizer identity, so the exporter also records the tokenizer's sha256 in the graph, and `OmniASR` checks it on load.

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

ONNX FP32 is the reference runtime; TensorRT FP32, FP16 and mixed are explicit, separate engines (not fallback-interchangeable), each gated per language against the FP32 baselines with `tools/fleurs_eval.py --backend tensorrt`.
</details>

## Performance and accuracy

Measured on the pinned 8-language FLEURS sample (`benchmarks/fleurs/manifest.json`, 160 test utterances) with `omniASR-CTC-300M-v2` and TensorRT 10.16 (details and every variant in [`benchmarks/tensorrt_fp16_sensitivity.json`](benchmarks/tensorrt_fp16_sensitivity.json)):

| TensorRT precision | Transcripts identical to FP32 | Frame argmax agreement | Latency vs FP32 | Per-language FP32 gate |
|---|---:|---:|---:|---|
| `fp32` | 160/160 (matches CPU ONNX exactly) | 100% | 1.0× | pass |
| `fp16` | 141/160 | 99.80% | ~0.19× | pass (WER within 0.5 points everywhere) |
| `mixed` | 150/160 | 99.91% | ~0.38× | pass |

Pinning layer groups to FP32 one at a time showed that only the convolutional front end is precision-sensitive under FP16: the 7-block waveform feature extractor and the positional convolution. Keeping them in FP32 (`precision="mixed"`) cuts the median per-clip logit error about 4× and leaves the remaining differences as near-tie token flips. Normalization, softmax, attention, residual adds, the final projection and every quarter of the 24 encoder blocks made no measurable difference. Regate any engine against the FP32 baselines on a GPU with:

```bash
python tools/fleurs_eval.py run --model ctc --backend tensorrt --precision mixed --gate
```

An earlier 40-clip English/Turkish experiment on an older TensorRT and profile reported 34/40 identical FP16 transcripts; those numbers are superseded by the measurements above.

## Benchmark and tests

```bash
python benchmarks/benchmark.py speech.wav --model dynamic.onnx --tokenizer tokenizer.model
python benchmarks/benchmark.py speech.wav --model omniasr_fp16.engine --tokenizer tokenizer.model --backend tensorrt

python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest tests/unit
```

Integration tests need real assets via env vars and skip otherwise: `OMNIASR_ONNX`/`OMNIASR_TOKENIZER` for ONNX, `OMNIASR_TRT_FP32_ENGINE` (+ the same tokenizer) for TensorRT, `OMNIASR_HUB_TESTS=1` for the Hub and `from_pretrained` TensorRT tests. They use the committed English FLEURS clip unless `OMNIASR_TEST_WAV` names another, and compare exact transcripts declared in `tests/fixtures/integration_expectations.json`; local assets on the default clip name their published model with `OMNIASR_TEST_MODEL` or give `OMNIASR_TEST_TRANSCRIPT` (see [CONTRIBUTING.md](CONTRIBUTING.md#test-tiers)). `tests/integration/test_tensorrt_fleurs.py` (GPU, `OMNIASR_REAL_MODELS=1`) builds `fp32`, `fp16` and `mixed` engines from the published model and checks them on the FLEURS fixtures. `tests/integration/test_hub.py` hits the real Hub repo over the network.

`tests/integration/test_real_models_cpu.py` runs the pinned published CTC and LLM 300M v2 models end to end on CPU against three committed [FLEURS](https://huggingface.co/datasets/google/fleurs) clips (English, Turkish, Central Kurdish; CC BY 4.0, see `tests/fixtures/fleurs/`), checking per-clip CER ceilings, determinism and clean shutdown. The ceilings are each model's measured CER plus 8 points: a smoke signal that catches a broken model or runtime, not small accuracy regressions. Per-language accuracy is gated separately: `tools/fleurs_eval.py` scores both models on a pinned 8-language FLEURS sample and fails on regressions past `benchmarks/fleurs/thresholds.json` (see [benchmarks/fleurs](benchmarks/fleurs/README.md)); the same workflow runs it. To gate a release, dispatch that workflow on the release commit before tagging and require it to pass. It downloads about 8 GB, so it is opt-in locally (`OMNIASR_REAL_MODELS=1 pytest tests/integration/test_real_models_cpu.py`) and runs in the `Real-model CPU validation` workflow weekly, after each release is published (reporting, not gating), on demand, and on PRs that change it or the runtime code it exercises. `tests/integration/test_real_models_edge_cases.py` does the same for clips derived from them (no speech, clipping, long pauses, code-switching, repetition and the 400-sample and 30-second limits), checking output and LLM stop reasons; see `tests/fixtures/edge_cases/README.md`.

## LLM-ASR models

`fast_omniasr.llm.OmniASRLLM` runs the wav2vec2 encoder + LLaMA decoder LLM-ASR models with
KV-cached greedy decoding on ONNX Runtime (CPU or CUDA). Inference imports only NumPy, ONNX
Runtime and SentencePiece.

```python
from fast_omniasr.llm import OmniASRLLM

model = OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-1B-v2-ONNX", device="cuda")
result = model.transcribe("speech.wav", lang="eng_Latn")
print(result.text, result.stop_reason)  # "eos" means complete
```

`from_pretrained` downloads every file listed in the repo's `config.json` and verifies its
SHA-256. `precision="fp16"` or `"bf16"` loads the half-precision variant (see
[Precision](#precision-fp32-fp16-bf16)). `lang` takes an upstream code such as `eng_Latn`, `tur_Latn` or `ckb_Arab` (see the
repo's `languages.json`). Omitting it uses the model's learned "no language" embedding, as
upstream does, and warns: upstream notes quality can degrade, and in our tests the original
(v1) models transcribed English as looping Arabic text without a language code.

| Model | Repository | Steps checked | Max logit error vs fairseq2 (CPU / CUDA) |
|---|---|---:|---:|
| `omniASR_LLM_300M` | `EmreAkgul/omniASR-LLM-300M-ONNX` | 885 | 4.4e-5 / 9.2e-5 |
| `omniASR_LLM_1B` | `EmreAkgul/omniASR-LLM-1B-ONNX` | 756 | 1.2e-4 / 8.5e-5 |
| `omniASR_LLM_3B` | `EmreAkgul/omniASR-LLM-3B-ONNX` | 848 | 7.8e-5 / 1.7e-4 |
| `omniASR_LLM_7B` | `EmreAkgul/omniASR-LLM-7B-ONNX` | 665 | 5.1e-4 / 1.5e-3 |
| `omniASR_LLM_300M_v2` | `EmreAkgul/omniASR-LLM-300M-v2-ONNX` | 772 | 5.1e-5 / 7.5e-5 |
| `omniASR_LLM_1B_v2` | `EmreAkgul/omniASR-LLM-1B-v2-ONNX` | 774 | 2.6e-3 / 4.4e-4 |
| `omniASR_LLM_3B_v2` | `EmreAkgul/omniASR-LLM-3B-v2-ONNX` | 772 | 1.2e-4 / 1.3e-4 |
| `omniASR_LLM_7B_v2` | `EmreAkgul/omniASR-LLM-7B-v2-ONNX` | 770 | 9.3e-5 / 1.4e-4 |

Each model was decoded on English, Turkish and Central Kurdish clips with and without
language conditioning (7B v1 never reaches EOS on English without a language code, in
fairseq2 as well, so that run is omitted). Every step's logits were compared with fairseq2 0.6
incremental greedy decoding on CPU float32, with PyTorch and fairseq2 imports blocked in the
ONNX process: greedy tokens and text matched at every step through EOS on CPU and CUDA. The
ONNX encoder is additionally checked against a float64 fairseq2 reference; its error must
stay within 3x PyTorch float32's own error. Reports are published under each repo's
`parity/`.

The exported encoder folds the positional convolution's weight normalization into a single
float64-computed weight. Recomputing it at runtime in float32 made ORT's encoder 12-37x less
accurate than PyTorch against float64 on the 3B model.

Sessions stay loaded across `transcribe()` calls and, on CUDA, the K/V cache stays on the
GPU between decoding steps (I/O binding); only token ids and logits cross to the host. On a
data-center GPU this decodes a 10-20 s clip in about 1.5-2 s (11-16 ms per token) with the 300M and 7B v2
models after the first call, which loads the weights. One `decoder.onnx` session serves the
prefill and every decoding step, so the GPU holds the encoder and a single copy of the decoder:
about 10 GiB for 300M (12 GiB for 30 s inputs) and 34 GiB for 7B v2. On smaller GPUs pass
`low_memory=True` to `from_pretrained` to load one session at a time (300M peaks at ~5.4 GiB),
at the cost of reloading weights for every clip.

Scope: batch one, clips up to 30 seconds, greedy decoding (upstream defaults to five-beam
search, so outputs can differ). The zero-shot (`7B_ZS`) and streaming `Unlimited` models use
other prompt formats and are not supported. Load only trusted model assets.

### Exporting and validating an LLM model

Export requires a working upstream `omnilingual_asr` / fairseq2 installation (tested with
fairseq2 0.6, omnilingual-asr 0.2.0 and PyTorch 2.8):

```bash
M=omniASR_LLM_1B_v2; CKPT=omniASR-LLM-1B-v2.pt; D=artifacts/1b-v2
python export_llm.py --model $M --checkpoint $CKPT --output $D/ts --audio english.wav
python tools/export_cached_llm.py $D/ts --audio turkish.wav --audio kurdish.wav
python export_llm_onnx.py $D/ts $D/onnx --stage encoder
python export_llm_onnx.py $D/ts $D/onnx --stage decoder
python tools/validate_llm_encoder_fp64.py $D/ts $D/onnx english.wav turkish.wav --checkpoint $CKPT
python tools/export_llm_languages.py $D/onnx --checkpoint $CKPT
python tools/validate_llm_fairseq2.py reference --model $M --checkpoint $CKPT --output $D/ref \
  english.wav english.wav --lang none --lang eng_Latn
python tools/validate_llm_fairseq2.py compare $D/ref --directory $D/onnx --device cpu \
  --report $D/onnx/fairseq2_parity_cpu.json   # repeat with --device cuda
python tools/publish_llm_hub.py $D/onnx --license LICENSE   # add --upload to publish
```

`FAST_OMNIASR_THREADS` sets the thread count for these tools. After publishing,
`validate_llm_fairseq2.py compare --repo <repo id>` repeats the comparison through
`from_pretrained`. `tools/validate_llm_onnx.py` compares raw ONNX tensors with the TorchScript
graphs stage by stage; both sides are float32, so it is a diagnostic rather than an accuracy
gate on long or noisy input.

### TorchScript reference runtime

The exported TorchScript graphs remain available as a correctness baseline (CPU float32,
batch one, clips up to 30 seconds), with PyTorch as the only extra dependency
(`pip install "fast-omniasr[llm]"`):

```python
from fast_omniasr.llm import OmniASRLLM

model = OmniASRLLM("/path/to/llm-300m-v2")               # full-prefix decoder
model = OmniASRLLM("/path/to/llm-300m-v2", cached=True)  # explicit KV-cache decoder
result = model.transcribe("speech.wav", max_new_tokens=512)
print(result.text, result.stop_reason)  # "eos" means complete
```

Building such a directory (requires fairseq2 and omnilingual_asr):

```bash
python export_llm.py --output DIR --audio speech.wav       # encoder.pt, decoder.pt, tokenizer
python tools/export_cached_llm.py DIR --audio speech.wav  # adds decoder_cached.pt (cached=True)
python tools/export_llm_languages.py DIR [--checkpoint CKPT]  # adds language tables (lang=...)
python tools/validate_llm.py DIR [--cached]                # re-checks with reference packages blocked
```

Without the language tables, `transcribe(..., lang=...)` raises; without `lang` the model
uses its learned "no language" embedding and warns, as with the ONNX runtime. The real-artifact integration test runs with
`FAST_OMNIASR_LLM_DIR=DIR pytest tests/integration/test_llm.py` (`FAST_OMNIASR_LLM_TEST_WAV`
overrides the clip).

How the TorchScript, explicit-cache and ONNX exports were built and validated, with the
parity reports from each stage, is recorded in the
[LLM export history](docs/llm-export-history.md) (historical developer notes, not current
behaviour).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the test tiers, reproducing the benchmarks, and the pull-request and release process. Pull requests target the `develop` branch.

## License

Project code is [MIT](LICENSE) and does not bundle any model weights. Converted ONNX/tokenizer assets are published separately under Apache-2.0 for [300M](https://huggingface.co/EmreAkgul/omniASR-CTC-300M-ONNX), [300M v2](https://huggingface.co/EmreAkgul/omniASR-CTC-300M-v2-ONNX), [1B](https://huggingface.co/EmreAkgul/omniASR-CTC-1B-ONNX), [1B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-1B-v2-ONNX), [3B](https://huggingface.co/EmreAkgul/omniASR-CTC-3B-ONNX), [3B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-3B-v2-ONNX), [7B](https://huggingface.co/EmreAkgul/omniASR-CTC-7B-ONNX), and [7B v2](https://huggingface.co/EmreAkgul/omniASR-CTC-7B-v2-ONNX), per the upstream [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) license (format conversion only, no retraining). Not affiliated with or endorsed by Meta or NVIDIA.
