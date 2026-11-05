# Long audio

`transcribe_long` cuts a recording at fixed intervals, transcribes each chunk on its own and
joins the text in order. Chunks don't overlap and nothing is reconciled across a cut. `.chunks`
gives each chunk's start and end in seconds and its text, plus its stop reason on the LLM.

faster-whisper's batched pipeline also transcribes chunks independently with no overlap, but it
cuts at silence using VAD. By default cuts here are fixed, so audio can be cut mid-word and a word
on a boundary can be damaged. That is the main cost of this approach.

`vad_filter=True` works like that batched pipeline, with the same Silero model and defaults: it
finds speech, joins it into chunks of at most `chunk_length` seconds of audio with the silence left
out, and keeps each chunk's start and end as times in the original recording, so `end - start` can
exceed `chunk_length`. When the VAD finds no speech, the decoder doesn't run and the result has empty
text and no chunks. That stops the LLM inventing text from silence and plain noise. Non-speech the
VAD mistakes for speech, such as tones or reversed speech in our tests, still reaches the decoder,
and the LLM can still invent text from it.

It needs `onnxruntime`, which the `onnx` extra installs, and runs on CPU. It's off by default, so
nothing changes unless you ask.

Text is joined with a space, including in scripts written without spaces. On TensorRT, each
chunk must fit the engine's sample profile. A chunk outside it fails when it's reached, which with
fixed cuts is the first chunk for one that's too long and the last for one that's too short, and
with VAD can be any chunk, after the earlier ones have decoded.

## Why 15 seconds

Whisper's 30 seconds comes from its fixed input window. OmniASR has no such window, so
`chunk_length` defaults to a value measured on these models, and accuracy turned out to
depend on it strongly.

Each recording joins 20 FLEURS test utterances from one language, with half a second of
silence between them, every utterance at the same level, for 3 to 5 minutes in total. It's
cut at each chunk length and scored against the joined references. "Alone" decodes each
utterance by itself, so it's the best case, with no cuts. Models are the pinned 300M v2.

CTC word error rate (%):

| language | alone | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| German | 13.0 | 19.4 | 19.9 | 15.4 | 20.1 |
| English | 19.5 | 23.4 | 24.6 | 21.1 | 23.7 |
| Spanish | 9.3 | 13.2 | 15.0 | 14.4 | 23.1 |
| Hindi | 12.7 | 20.0 | 18.6 | 27.4 | 33.3 |
| Turkish | 21.7 | 30.7 | 29.9 | 34.5 | 42.1 |
| French | 18.3 | 23.7 | 22.6 | 22.1 | 22.6 |
| mean | 15.7 | 21.7 | 21.8 | 22.5 | 27.5 |

No length wins everywhere: 20 s is the best tested on German, English and French, 10 s on
Spanish, and 15 s on Turkish, while Hindi did best at 5 s (17.8). 10 s and 15 s tie for the best
mean. 20 s and above bring large losses on Hindi and Turkish. Other lengths weren't tested on
Turkish and French.

On these recordings, some languages failed badly on long chunks: Spanish reached 40.9 and Hindi
56.4 at 40 s, and English 59.9 at 60 s, with whole spans of speech missing. Spanish still degraded
when chunks were whole utterances, so cuts through speech aren't the cause. The cause is unknown.
It may be the recordings themselves, which change speaker every few seconds: in a small check, four
Spanish utterances, each repeated up to about a minute, stayed roughly stable. Don't read this as a limit on
continuous speech from one speaker, which wasn't measured.

With `vad_filter=True`, on the same recordings:

| language | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|
| German | 20.3 | 15.7 | 17.4 | 18.4 |
| English | 22.3 | 22.0 | 22.0 | 23.7 |
| Spanish | 10.1 | 9.9 | 10.1 | 9.9 |
| Hindi | 24.5 | 22.5 | 20.8 | 21.6 |
| Turkish | 25.5 | 24.2 | 25.0 | 24.2 |
| French | 19.9 | 20.3 | 21.0 | 21.4 |
| mean | 20.4 | 19.1 | 19.4 | 19.9 |

VAD beat fixed cuts at every length on the mean and removed the drop at 30 seconds. It isn't better
everywhere: Hindi is worst, about 4 points higher at 10 and 15 seconds, German is higher at 10 and 20
seconds, and English is higher at 20 and level at 30. These recordings also have clean
half-second gaps between sentences, which is an easy case for VAD, so expect less on natural speech.

LLM word error rate (%), with fixed cuts:

| language | alone | 5 s | 10 s | 15 s | 20 s | 25 s | 30 s |
|---|---|---|---|---|---|---|---|
| English | 10.7 | 17.6 | 14.2 | 10.9 | 14.9 | 10.4 | 10.2 |
| Turkish | 8.4 | - | - | 12.5 | 11.4 | - | 12.2 |

15 seconds is a safe choice, not a proven best. It ties for the best CTC mean, avoids the large
losses at 20 s and above, is close to the alone case on English on the LLM and within a point or so of the other lengths on Turkish, and is the segment length of upstream's
`Unlimited` LLM models. Differences of a point or so here aren't meaningful: moving where the
cuts land changed German at 10 s by 1.5 points, and the LLM series above has no clear trend. The
recordings are small, one language each, and change speaker every few seconds, so these numbers
compare lengths on this audio and don't predict accuracy. Pass `chunk_length` to use another
value; the LLM's limit is 30 seconds.

Upstream's `Unlimited` cards decode long audio natively, but this runtime doesn't support them.
