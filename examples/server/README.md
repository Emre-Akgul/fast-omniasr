# Deployment example

A minimal HTTP service around one preloaded `OmniASR` model. `server.py` uses only the
standard library on top of fast-omniasr, so serving adds no dependencies. Use it as a
starting point, not as a hardened production server: put it behind your own proxy for TLS,
authentication and request timeouts.

```bash
docker build -f examples/server/Dockerfile -t fast-omniasr-server .   # from the repo root
docker run --rm -p 8000:8000 fast-omniasr-server
curl --data-binary @speech.wav http://localhost:8000/transcribe
# {"text": "...", "duration": 4.32}
```

The image downloads the model at build time (`--build-arg MODEL=...`, `PRECISION=...`) and
starts with `--local-files-only`, so it needs no network at run time. For a GPU, build with
`--build-arg EXTRAS=cuda,hub,resample` and run with `--gpus all -e DEVICE=cuda`.
Without Docker: `python examples/server/server.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX`.

| Endpoint | Meaning |
|---|---|
| `GET /health` | 200 while the process is up (liveness) |
| `GET /ready` | 200 once the model is loaded; 503 while loading and during shutdown (readiness) |
| `POST /transcribe` | body: an audio file (WAV, FLAC, ...; any rate or channel count) |

| Situation | Response |
|---|---|
| Body over `--max-bytes` (default 20 MiB) | 413, body not read |
| Audio over `--max-seconds` (default 120) | 413 |
| Audio that can't be decoded | 400 |
| `--workers` running and `--max-queue` waiting (defaults 1 and 8) | 503 with `Retry-After: 1` |
| Not ready yet, or shutting down | 503 with `Retry-After: 1` |

Audio longer than one chunk is transcribed with `transcribe_long`. One model instance serves
every worker. Extra workers let requests overlap on the same ONNX Runtime session, so on a
CPU, set `--workers` and the model's threads so that their product fits your cores. For more
GPU throughput run more containers.

On SIGTERM (`docker stop`) the server marks itself not ready, stops accepting connections,
finishes the requests in flight, closes the model and exits 0. Give `docker stop -t` more
time than your longest request. The image's `HEALTHCHECK` polls `/ready`.

`tests/unit/test_server_example.py` covers the limits, queueing and shutdown, and
`.github/workflows/server-example.yml` builds the image and checks a real transcript.
