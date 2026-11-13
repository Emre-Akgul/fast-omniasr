# CTC timestamps

`word_timestamps=True` on `transcribe`, `transcribe_numpy` and `transcribe_long` sets `.words` and `.token_times` on the result, and on each of `.chunks`. It uses the same frames as plain decoding, so `text` and `token_ids` don't change. The LLM models don't have this: they have no frames to read times from.

## How times are computed

The model emits one frame per 320 samples, or 20 ms, measured on `omniASR-CTC-300M-v2`. The frame count for `n` samples is `(n - 400) // 320 + 1`. Frame `i` is given the time `i * 0.02`. If a model's frame count doesn't match that formula, `word_timestamps=True` raises `ValueError` rather than return wrong times.

Greedy decoding collapses repeated frames into tokens and drops blanks. Each token keeps the frames it collapsed from: `start` is the time of its first frame and `end` the time of the frame after its last. `token_times` has one `(start, end)` for each entry in `token_ids`.

## Words

A word ends at a whitespace token and starts at a token with leading whitespace. Whitespace tokens belong to no word, though `token_times` still has their times. Control tokens are left out. A `Word` has `start`, `end` and `word`, where `start` is that of its first token and `end` that of its last, apart from the removed gaps described under Limits. `<unk>` decodes to ` ⁇ `, so a word that holds it alongside other tokens contains spaces.

Text written without spaces, such as Chinese, Japanese or Thai, comes out as one long word, and faster-whisper's per-character split for those languages isn't applied, since CTC isn't told the language. Use `token_times` there.

## Limits

- **Where a time falls in a word:** CTC emits a token at any frame it likes within the sound, often a short spike. A word's `start` and `end` cover the frames that emitted it, so they can fall inside the sound rather than at its edges, and adjacent words aren't made to touch.
- **Accuracy:** tested for consistency only. Adding silence before a clip moves words by the same amount to within 20 ms, and words stay in order inside the clip. They weren't compared with hand-labelled word boundaries, so absolute error isn't known.
- **No confidence:** times are alignment, not a measure of how sure the model is. There's no `probability` field.
- **Chunks:** in `transcribe_long` each chunk is decoded alone, so a word on a fixed cut comes out as two words or is damaged. With `vad_filter=True`, a chunk's audio has silence removed. Times are mapped back to the recording as faster-whisper does: a word goes with the speech span holding its midpoint, and a word that crosses a removed gap gets a single shift, so its times don't include the gap. Each token is mapped the same way on its own, so around a removed gap a word's times can differ from those of its first and last tokens.
