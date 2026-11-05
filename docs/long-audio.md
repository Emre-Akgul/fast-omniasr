# Long audio

`transcribe_long` cuts a recording at fixed intervals, transcribes each chunk on its own and
joins the text in order. Chunks don't overlap and nothing is reconciled across a cut, like
faster-whisper, which also cuts audio and joins the results. `.chunks` gives each chunk's
start and end in seconds and its text, plus its stop reason on the LLM.

Audio can be cut mid-word, so a word on a boundary can be damaged. That is the main cost of
chunking.

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
| mean | 15.8 | 21.7 | 21.6 | 22.5 | 27.7 |

Past 20 seconds, CTC gets much worse on some languages, and it isn't about the cuts: at 40
seconds Spanish reached 40.9 and Hindi 56.4, and at 60 seconds English reached 59.9. So an
unchunked call on long audio can be far worse than chunking, even on ONNX, which accepts any
length.

LLM word error rate (%) on English, the alone case being 10.7:

| 5 s | 10 s | 15 s | 20 s | 25 s | 30 s |
|---|---|---|---|---|---|
| 17.6 | 14.2 | 10.9 | 14.9 | 10.4 | 10.2 |

15 seconds is close to the best on every language tested, and it's the segment length of
upstream's `Unlimited` LLM models. These recordings are one language at a time and small, and
changing speakers every few seconds is harder than one person talking, so treat the numbers as
a comparison between lengths, not as expected accuracy. Pass `chunk_length` to use another
value; the LLM's limit is 30 seconds.

Upstream's `Unlimited` cards decode long audio natively, but this runtime doesn't support them.
