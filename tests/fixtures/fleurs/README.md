# FLEURS fixture clips

Three short clips from the **test** split of [FLEURS](https://huggingface.co/datasets/google/fleurs)
(Conneau et al., 2022), used by `tests/integration/test_real_models_cpu.py` and the
`real-models` workflow to run the published models end to end on CPU.

- Source: `google/fleurs` at commit `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; the exact
  tarball member, reference transcript and sha256 of each clip are listed in `manifest.json`.
- Modifications: the original 32-bit float, 16 kHz mono WAV was quantized to 16-bit PCM
  (round to nearest) and FLAC-encoded. Nothing else was changed (no trimming, resampling or
  normalization). Both models produce the same transcripts on the original and quantized audio.
- License: FLEURS © Google, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  These modified clips are redistributed under the same license, without warranties of any kind.

```bibtex
@article{fleurs2022arxiv,
  title   = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author  = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and
             Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal = {arXiv preprint arXiv:2205.12446},
  year    = {2022}
}
```
