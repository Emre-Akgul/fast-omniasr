# FLEURS per-language regression gates

`tools/fleurs_eval.py` scores the published models per language on a fixed sample of the
[FLEURS](https://huggingface.co/datasets/google/fleurs) test split (FLEURS © Google, CC BY 4.0)
and fails when a language regresses past its threshold. The `Real-model CPU validation`
workflow runs it for the CTC and LLM 300M v2 models (`fleurs-ctc` and `fleurs-llm` jobs).

| file | contents |
|---|---|
| `manifest.json` | FLEURS commit; per language: OmniASR language code, gated metric, and the sampled utterances (file name, reference transcript, duration, audio sha256) |
| `thresholds.json` | per model: pinned repo/revision, utterances per language, and per language the measured baseline and the `max_wer` / `max_cer` gates |

**Sample.** 8 languages × 20 test utterances (≤ 20 s each, one recording per sentence,
ordered by a hash of the file name): English, Turkish, Central Kurdish, German, Spanish,
French, Hindi and Mandarin. The CTC gate uses all 20; the LLM gate uses the first 5 per
language to keep the CPU job under an hour. Mandarin is not space-delimited, so only its CER
is gated.

**Metrics.** Corpus WER/CER after the normalization in `tools/evaluate_precisions.py` (NFKC,
lowercase, punctuation and zero-width characters removed).

**Thresholds.** `max_wer = baseline + max(3 points, 15%)`, `max_cer = baseline + max(1.5
points, 15%)`, where the baseline is the pinned model on CPU in fp32. The gate guards a fixed
model on fixed audio against runtime regressions; the margin absorbs CPU-dependent token
flips. Gating the previous-generation `omniASR-CTC-300M-ONNX` against the v2 thresholds fails
on English and Central Kurdish CER, so real quality drops are caught. These numbers are a
small-sample regression signal, not a benchmark of model quality.

```bash
python tools/fleurs_eval.py run --model ctc --gate --report fleurs-ctc.json
python tools/fleurs_eval.py run --model llm --gate --report fleurs-llm.json
python tools/fleurs_eval.py run --model ctc --repo <repo> --revision <sha>   # score another export
```

Only the listed utterances are extracted from each pinned language tarball (about 3.6 GB
across the 8 languages on first use), checked against the manifest's sha256, and kept under
`--audio-dir` (default `.cache/fleurs`) for later runs.

**Updating.** When a model is republished, update its `revision` in `thresholds.json`, run
the tool with `--report`, and set the new baseline and limits with the policy above in the
same PR, so the change in accuracy is reviewed explicitly. `build-manifest` regenerates the
sample for a new FLEURS commit or language list.
