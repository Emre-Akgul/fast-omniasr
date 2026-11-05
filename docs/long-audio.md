# Long audio

`transcribe_long` cuts a recording at fixed intervals, transcribes each chunk on its own and
joins the text in order. Chunks don't overlap and nothing is reconciled across a cut. `.chunks`
gives each chunk's start and end in seconds and its text, plus its stop reason on the LLM.

faster-whisper's batched pipeline also transcribes chunks independently with no overlap, but it
cuts at silence using VAD. Here cuts are fixed, so audio can be cut mid-word and a word on a
boundary can be damaged. That is the main cost of this approach.

Text is joined with a space, including in scripts written without spaces. On TensorRT, each
chunk must fit the engine's sample profile: a chunk above its maximum fails on the first chunk,
and a last chunk below a custom minimum fails after the others have decoded.

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

LLM word error rate (%) on English, the alone case being 10.7:

| 5 s | 10 s | 15 s | 20 s | 25 s | 30 s |
|---|---|---|---|---|---|
| 17.6 | 14.2 | 10.9 | 14.9 | 10.4 | 10.2 |

15 seconds is a safe choice, not a proven best. It ties for the best CTC mean, avoids the large
losses at 20 s and above, did well on English on the LLM, and is the segment length of upstream's
`Unlimited` LLM models. Differences of a point or so here aren't meaningful: moving where the
cuts land changed German at 10 s by 1.5 points, and the LLM series above has no clear trend. The
recordings are small, one language each, and change speaker every few seconds, so these numbers
compare lengths on this audio and don't predict accuracy. Pass `chunk_length` to use another
value; the LLM's limit is 30 seconds.

Upstream's `Unlimited` cards decode long audio natively, but this runtime doesn't support them.
