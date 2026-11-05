# Edge-case fixture clips

Clips for behaviour the FLEURS sentences don't reach: audio with no speech, damaged speech,
and the length limits. `clips.py` rebuilds them from `tests/fixtures/fleurs` and seeded noise,
so no extra audio is committed. They are modified FLEURS audio, under the same
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms as the source
(see `../fleurs/README.md`).

| clip | what it is |
|---|---|
| `silence`, `white_noise`, `mains_hum` | no speech |
| `clipped_speech` | speech with about 23% of samples at full scale |
| `long_pause` | eight seconds of room noise inside a sentence |
| `code_switching` | an English sentence, then a Turkish one, at the same level |
| `repeated_sentence`, `stutter` | a sentence four times; a quarter second looped for 20 s |
| `shortest_clip`, `longest_clip` | exactly 400 and exactly 480,000 samples |

`manifest.json` holds each clip's sha256 and what the pinned models do on it on CPU, with
`lang="eng_Latn"` for the LLM. `tests/unit/test_edge_case_fixtures.py` checks the clips, and
`tests/integration/test_real_models_edge_cases.py` and `test_tensorrt_fleurs.py` check the
models. Stop reasons apply to the LLM only: CTC has no generation loop.

These are recorded behaviour, not targets:

- The LLM invents text when there is no speech, and on `silence`, `mains_hum` and `stutter`
  it is still repeating itself at the 64-token budget these clips use. At the default 512,
  `silence` and `stutter` run out the budget and `mains_hum` reaches EOS after 210 tokens.
  The test pins the stop reason so a change is noticed, not because it's wanted. `silence` is
  digital zero: faint added noise changes the result, so it says nothing about room silence.
  `repetition_penalty=1.2` ends the loops on `silence` and `mains_hum`, not on `stutter`, and
  it cuts about 40% of the transcript on `longest_clip`, so it is no general fix.
- CTC returns empty text on all three no-speech clips, and repeated text on `stutter`, where
  only determinism is checked.
- `code_switching` has high error rates because the English half comes out garbled on both
  models, even at matched levels and with `lang` set. Only the end of the Turkish half is
  checked exactly.
- Text is compared by CER ceiling, not exactly. The ceilings are the measured CER plus some
  slack, for the models pinned in `../fleurs/manifest.json`; re-measure them with new models.

Quiet speech is deliberately not covered: nothing is expected of it.
