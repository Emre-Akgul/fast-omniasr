# Test matrix

fast-omniasr supports 16 published models (8 CTC, 8 LLM; original and v2; 300M to 7B),
three runtimes (ONNX Runtime, TensorRT for CTC, and the TorchScript LLM reference), CPU and
CUDA, and fp32/fp16/bf16 graphs (plus TensorRT's fp16 and `mixed` engines). Running every
combination on every change would mean downloading and running hundreds of gigabytes of
weights, so testing is tiered:

- **unit** checks every configuration's code path on every change, with fakes and tiny
  generated graphs.
- **real-cpu** runs one pinned model per family end to end on hosted CPUs, on a schedule,
  for releases and for runtime changes.
- **gpu** is the manual CUDA/TensorRT pass, since there is no hosted GPU runner.
- **release** runs once per model when it is exported or republished, and its report is
  published with the model.

The cards within a family share the runtime code paths and differ in encoder size,
vocabulary and tokenizer, so the pinned model's tiers stand in for the others between
releases. The source of truth is [`tests/matrix.json`](../tests/matrix.json); the tables
below are generated from it (`python tools/test_matrix.py --write`), and a unit test keeps
them, the model cards, backends and precisions consistent.

Known gaps, by design of the tiers: CPU half-precision graphs are covered only by unit
tests (the release reports measure precision agreement on CUDA), CTC exports have no
release-time comparison against the source model yet (tracked separately), and TensorRT is
tested only for the pinned CTC model.

<!-- matrix:begin -->
### Tiers

| tier | when | what runs | command |
|---|---|---|---|
| `unit` | every push and PR (Lint and CPU unit tests; Package) | pytest tests/unit, against fakes and tiny generated graphs; no real weights | `python -m pytest tests/unit` |
| `real-cpu` | weekly, on each published release, on demand, and on PRs touching the runtime (Real-model CPU validation) | the pinned published model end to end on CPU (FLEURS fixtures, CER ceilings, determinism) plus the per-language FLEURS regression gate | `OMNIASR_REAL_MODELS=1 python -m pytest tests/integration/test_real_models_cpu.py && python tools/fleurs_eval.py run --model {family} --gate` |
| `gpu` | manually on a CUDA machine, before merging changes to GPU, precision, TensorRT or export code (no hosted GPU runner) | ONNX CUDA precision agreement and TensorRT engines of the pinned model on the FLEURS fixtures and gate | `python tools/evaluate_precisions.py --{family} <export> --device cuda --precision fp32 fp16 bf16 --dataset <eval.jsonl>; OMNIASR_REAL_MODELS=1 python -m pytest tests/integration/test_tensorrt_fleurs.py; python tools/fleurs_eval.py run --model ctc --backend tensorrt --precision mixed --gate` |
| `release` | once per model, when it is exported or republished | the precision-agreement report published in the model card (measured with CUDA; tools/publish_precisions.py); for LLMs, step-by-step fp32 logit parity with fairseq2 on CPU and CUDA (tools/validate_llm_fairseq2.py) | `python tools/evaluate_precisions.py --{family} <export> --device cuda --precision fp32 fp16 bf16 --audio <clips> --report <report>; python tools/publish_precisions.py {repo} <export> --report <report> [--upload]; for LLMs also python tools/validate_llm_fairseq2.py compare <reference> --directory <export> --device cpu|cuda` |

### CTC models

`omniASR_CTC_300M_v2` is the pinned model the scheduled tiers run; the other 7 cards (`omniASR_CTC_300M`, `omniASR_CTC_1B`, `omniASR_CTC_3B`, `omniASR_CTC_7B`, `omniASR_CTC_1B_v2`, `omniASR_CTC_3B_v2`, `omniASR_CTC_7B_v2`) share its code paths.

| backend | device | precision | `omniASR_CTC_300M_v2` | other cards |
|---|---|---|---|---|
| onnx | cpu | fp32 | unit, real-cpu | unit |
| onnx | cpu | fp16 | unit | unit |
| onnx | cuda | fp32 | unit, gpu, release | unit, release |
| onnx | cuda | fp16 | unit, gpu, release | unit, release |
| onnx | cuda | bf16 | unit, gpu, release | unit, release |
| tensorrt | cuda | fp32 | unit, gpu | unit |
| tensorrt | cuda | fp16 | unit, gpu | unit |
| tensorrt | cuda | mixed | unit, gpu | unit |

### LLM models

`omniASR_LLM_300M_v2` is the pinned model the scheduled tiers run; the other 7 cards (`omniASR_LLM_300M`, `omniASR_LLM_1B`, `omniASR_LLM_3B`, `omniASR_LLM_7B`, `omniASR_LLM_1B_v2`, `omniASR_LLM_3B_v2`, `omniASR_LLM_7B_v2`) share its code paths.

| backend | device | precision | `omniASR_LLM_300M_v2` | other cards |
|---|---|---|---|---|
| onnx | cpu | fp32 | unit, real-cpu, release | unit, release |
| onnx | cpu | fp16 | unit | unit |
| onnx | cuda | fp32 | unit, gpu, release | unit, release |
| onnx | cuda | fp16 | unit, gpu, release | unit, release |
| onnx | cuda | bf16 | unit, gpu, release | unit, release |
| torch | cpu | fp32 | unit | unit |
<!-- matrix:end -->

To run a tier for a model other than the pinned one, point the same commands at it: the
real-model tests read `OMNIASR_CTC_REPO`/`OMNIASR_CTC_REVISION` and
`OMNIASR_LLM_REPO`/`OMNIASR_LLM_REVISION`, and `tools/fleurs_eval.py` takes `--repo` and
`--revision` (its gate's thresholds are only defined for the pinned models).
