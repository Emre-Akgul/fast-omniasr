# Contributing to fast-omniasr

Thanks for helping. This guide covers setting up a development environment, the test
tiers, reproducing benchmarks, and what a pull request needs before it is merged.

## Branches and pull requests

- `main` holds released code; day-to-day work lands on **`develop`**.
- Branch from `develop` (`fix/issue-123-short-name` or `feature/short-name`) and open the
  pull request **against `develop`**. Pull requests are squash-merged.
- One issue per pull request. Reference it with `Closes #123` in the description.
  GitHub only auto-closes issues on the default branch (`main`), so a maintainer closes
  the issue when the pull request is merged into `develop`.
- Keep changes focused: a bug fix doesn't need a drive-by refactor.

## Development setup

Python 3.10 or newer (CI tests 3.10–3.12).

```bash
git clone https://github.com/Emre-Akgul/fast-omniasr && cd fast-omniasr
git switch develop
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev,resample]"   # lint + unit tests, CPU only
python -m pip install -e ".[dev,onnx,hub]"   # + ONNX Runtime CPU and Hub downloads
```

Optional extras, as needed:

| extra | for |
|---|---|
| `onnx` / `cuda` | ONNX Runtime CPU / CUDA (`onnxruntime-gpu`) |
| `hub` | `from_pretrained` downloads (`huggingface_hub`) |
| `resample` | `convert_audio=True` for audio that isn't 16 kHz (`soxr`) |
| `tensorrt` | the TensorRT backend (`cuda-python`; TensorRT itself is installed separately) |
| `llm` | the TorchScript LLM baseline and LLM tests (`torch`) |

Exporting models additionally needs `fairseq2`/`omnilingual-asr` and `onnx`; see
"Export and engine building" in the README.

## Test tiers

| tier | command | needs | runs in CI |
|---|---|---|---|
| Lint | `python -m ruff check .` | `dev` | every push and PR |
| Unit | `python -m pytest tests/unit` | `dev,resample` (CPU only, no models, no network; resampling tests skip without `resample`) | every push and PR, Python 3.10/3.11/3.12; cache locking also on Windows |
| Package | `python -m build`, then install the wheel in a clean venv and run `python tests/packaging/smoke.py <extra>` and `pytest tests/unit` | `build` | every push and PR (`Package` workflow): the wheel's unit suite on Linux, Windows and macOS with Python 3.10 and 3.12, each extra (`onnx`, `hub`, `llm`, `tensorrt`, `resample`) installed alone, and the oldest declared dependency versions (`tests/packaging/minimum-constraints.txt`) |
| LLM unit | `python -m pytest tests/unit/test_llm.py tests/unit/test_llm_cache.py tests/unit/test_llm_onnx.py` | `dev,llm,onnx` extras and the `onnx` package | every push and PR |
| Real models, CPU | `OMNIASR_REAL_MODELS=1 python -m pytest tests/integration/test_real_models_cpu.py tests/integration/test_real_models_edge_cases.py` | `dev,onnx,hub,resample`; downloads ~8 GB | `Real-model CPU validation`: weekly, after each published release, on demand, and on PRs that change the paths listed in `real-models.yml` (the ONNX/LLM runtime, hub, audio and evaluation code; not the TensorRT backend) |
| FLEURS gates | `python tools/fleurs_eval.py run --model ctc --gate` (and `--model llm`) | `dev,onnx,hub` | same workflow (`fleurs-ctc`, `fleurs-llm`) |
| GPU / TensorRT | `OMNIASR_REAL_MODELS=1 python -m pytest tests/integration/test_tensorrt_fleurs.py`, plus `test_tensorrt*.py`/`test_onnx.py` with the `OMNIASR_*` variables below | a CUDA GPU, `cuda`/`tensorrt` | not in CI: run locally for GPU changes |

Which tier covers which model, backend, device and precision is laid out in the
[test matrix](docs/test-matrix.md) (source: `tests/matrix.json`; regenerate the page with
`python tools/test_matrix.py --write` after editing it).

Unit tests must not need a GPU, model downloads or the network. Integration tests skip
unless their inputs are provided:

- `OMNIASR_ONNX`, `OMNIASR_TOKENIZER`: a local ONNX export (`test_onnx.py`)
- `OMNIASR_TRT_FP32_ENGINE`: a TensorRT engine, with the tokenizer above
  (`test_tensorrt.py`); length checks follow the engine's own profile
- `OMNIASR_HUB_TESTS=1`, plus network access: `test_hub.py` and
  `test_tensorrt_from_pretrained.py` download the published model (the latter checks
  both the default 400–480,000-sample profile and a custom one)
- `OMNIASR_TEST_WAV` (optional): the clip; by default the committed English FLEURS clip.
  Exact transcripts come from `tests/fixtures/integration_expectations.json` (matched by
  the clip's sha256 and `<repo>@<commit>`). On the default clip a transcript is required:
  local assets set `OMNIASR_TEST_MODEL` to the published export they are (e.g.
  `EmreAkgul/omniASR-CTC-300M-v2-ONNX@6c1096d…`, the full commit) or
  `OMNIASR_TEST_TRANSCRIPT` to the expected text (e.g. for a fine-tuned export). With
  another clip and neither variable, exact-text checks are skipped
- `OMNIASR_REAL_MODELS=1`: download the pinned published models (`OMNIASR_CACHE_DIR` to
  choose the cache; `OMNIASR_CTC_REPO`/`OMNIASR_CTC_REVISION` and
  `OMNIASR_LLM_REPO`/`OMNIASR_LLM_REVISION` to test another repo or revision)
- `FAST_OMNIASR_LLM_DIR`, `FAST_OMNIASR_LLM_TEST_WAV`: a local LLM export

Changes to runtime code (`src/fast_omniasr/**`) need the unit tier green and, for
decoding, precision, GPU or export changes, the relevant real-model run. Say in the pull
request which tiers you ran and which you could not (e.g. no GPU available).

## Reproducing benchmarks and accuracy numbers

- **Per-language accuracy** (the gated numbers in `benchmarks/fleurs/thresholds.json`):
  `python tools/fleurs_eval.py run --model ctc --report fleurs-ctc.json` and
  `--model llm`. The sample is pinned (FLEURS commit and audio checksums in
  `benchmarks/fleurs/manifest.json`); see [benchmarks/fleurs](benchmarks/fleurs/README.md).
- **TensorRT precision** (the README's "Performance and accuracy" table):
  `python tools/fleurs_eval.py run --model ctc --backend tensorrt --precision mixed --gate`
  on a GPU, repeated for `fp32` and `fp16`.
- **Latency**: `python benchmarks/benchmark.py speech.wav --repo <repo> --precision fp16
  --device cuda --runs 20 --output report.json` (or `--model`/`--tokenizer` for local
  assets, `--backend tensorrt` for an engine). The report records the model identity,
  settings, versions, latency percentiles, real-time factor and peak memory; commit it with
  the numbers it supports.
- **Precision variants** (fp32/fp16/bf16 agreement, WER/CER and peak GPU memory):
  `python tools/evaluate_precisions.py --ctc <export dir> --dataset eval.jsonl` (JSONL
  manifest format in the tool's docstring).
- LLM cache and parity results: [benchmarks/README.md](benchmarks/README.md).

When you publish new numbers, commit the JSON report next to the claim and state the
model revision, precision, backend, library versions and sample. Describe hardware
generically (e.g. "a consumer GPU") rather than naming a specific machine.

## Pull request checklist

- [ ] Targets `develop` and references its issue (`Closes #123`)
- [ ] `ruff check .` and `pytest tests/unit` pass locally
- [ ] New behaviour has tests; bug fixes have a regression test that fails without the fix
- [ ] Public behaviour changes are reflected in the README or docstrings
- [ ] Tiers run and not run are listed; accuracy- or speed-affecting changes include
      before/after numbers
- [ ] No large binaries; test fixtures are small and licensed (see `tests/fixtures/fleurs`)

Every pull request gets a review before merge, and all required checks must pass.

## Releases

Maintainers release from `main`:

1. Merge `develop` into `main` once its CI passes, then dispatch the `Real-model CPU
   validation` workflow on `main` and require it to pass before tagging.
2. Bump `version` in `pyproject.toml`, tag `vX.Y.Z` and publish the GitHub release.
3. Build and upload the package from the tag: `python -m pip install build twine`,
   `python -m build`, then `python -m twine upload dist/*`.
4. If exported model assets changed, republish them with the tools under `tools/`
   (`publish_hub.py`, `publish_llm_hub.py`, and `publish_precisions.py` for fp16/bf16
   variants) so each `config.json` carries fresh checksums.

Before 1.0, a minor version (0.x) may change public behaviour; the release notes (the
GitHub release description) list every such change. Patch versions (0.x.y) only fix bugs.

## Reporting bugs and security issues

Use the issue templates. For a suspected security problem (for example in model
downloading or file handling), don't open a public issue: contact the maintainer
through the email on their GitHub profile.
