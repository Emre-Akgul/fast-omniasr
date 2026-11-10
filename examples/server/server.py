"""A minimal fast-omniasr HTTP service: standard library only, one model loaded at start-up.

    python examples/server/server.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --port 8000
    curl --data-binary @speech.wav http://localhost:8000/transcribe

Endpoints:
- `GET /health`: 200 while the process is up (liveness).
- `GET /ready`: 200 once the model is loaded, 503 before that and while shutting down.
- `POST /transcribe`: the request body is an audio file (WAV, FLAC, ...). Responds with
  `{"text": ..., "duration": seconds}`. 413 when the body or the audio is over the limit,
  400 for audio that can't be decoded, 415 for a sample rate other than 16 kHz when the
  `resample` extra isn't installed, 503 with `Retry-After` when the queue is full.

`--workers` requests run at once (one model instance is shared; the ONNX Runtime session
is thread-safe), at most `--max-queue` more wait, and the rest are refused right away
instead of piling up. SIGTERM or SIGINT stops accepting connections, lets in-flight requests
finish, then closes the model.
"""

import argparse
import io
import json
import logging
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("fast_omniasr.server")


class Limiter:
    """Admits `workers` requests at once and lets at most `max_queue` more wait."""

    def __init__(self, workers: int, max_queue: int):
        self.running = threading.BoundedSemaphore(workers)
        self.admitted = threading.BoundedSemaphore(workers + max_queue)

    def __enter__(self):
        if not self.admitted.acquire(blocking=False):
            raise QueueFull
        self.running.acquire()
        return self

    def __exit__(self, *exc_info):
        self.running.release()
        self.admitted.release()


class QueueFull(Exception):
    pass


class Service:
    def __init__(self, model, *, max_bytes: int, max_seconds: float, workers: int,
                 max_queue: int):
        self.model = model
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        self.limiter = Limiter(workers, max_queue)
        self.ready = threading.Event()

    def transcribe(self, body: bytes) -> dict:
        import soundfile as sf

        try:
            waveform, rate = sf.read(io.BytesIO(body), dtype="float32")
        except Exception as exc:  # libsndfile raises several types
            raise BadRequest(f"could not decode audio: {exc}") from exc
        duration = len(waveform) / rate
        if duration > self.max_seconds:
            raise TooLarge(f"audio is {duration:.1f} s; the limit is {self.max_seconds:g} s")
        with self.limiter:
            result = self.model.transcribe_long(waveform, rate, convert_audio=True)
        return {"text": result.text, "duration": round(duration, 3)}


class BadRequest(Exception):
    pass


class TooLarge(Exception):
    pass


def handler_for(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: int, body: dict, headers: dict | None = None) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"status": "ok"})
            elif self.path == "/ready":
                ready = service.ready.is_set()
                self._reply(200 if ready else 503, {"ready": ready})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/transcribe":
                self._reply(404, {"error": "not found"})
                return
            if not service.ready.is_set():
                self._reply(503, {"error": "not ready"}, {"Retry-After": "1"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._reply(411, {"error": "Content-Length is required"})
                return
            if length > service.max_bytes:
                # Don't read the body; close so the client doesn't keep sending it.
                self.close_connection = True
                self._reply(413, {"error": f"body is over {service.max_bytes} bytes"})
                return
            body = self.rfile.read(length)
            try:
                self._reply(200, service.transcribe(body))
            except QueueFull:
                self._reply(503, {"error": "busy"}, {"Retry-After": "1"})
            except TooLarge as exc:
                self._reply(413, {"error": str(exc)})
            except ImportError as exc:  # another sample rate without the resample extra
                self._reply(415, {"error": str(exc)})
            except (BadRequest, ValueError) as exc:
                self._reply(400, {"error": str(exc)})
            except Exception:  # report it and keep serving
                log.exception("transcription failed")
                self._reply(500, {"error": "transcription failed"})

        def log_message(self, format, *args):  # BaseHTTPRequestHandler names it format
            log.info("%s %s", self.address_string(), format % args)

    return Handler


class Server(ThreadingHTTPServer):
    # Join request threads on server_close(), so shutdown waits for in-flight requests.
    daemon_threads = False
    block_on_close = True


def serve(service: Service, host: str, port: int) -> Server:
    return Server((host, port), handler_for(service))


def stop(server: Server, service: Service) -> None:
    """Refuse new work, stop accepting, wait for in-flight requests, close the model."""
    service.ready.clear()
    server.shutdown()      # returns once serve_forever() has stopped accepting
    server.server_close()  # joins the request threads
    service.model.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", help="fp32 (default), fp16 or bf16")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true",
                        help="load the copy cached at build time; never contact the Hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="concurrent transcriptions")
    parser.add_argument("--max-queue", type=int, default=8, help="requests allowed to wait")
    parser.add_argument("--max-bytes", type=int, default=20 * 2**20, help="default 20 MiB")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="default 120")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_queue < 0 or args.max_bytes < 1 or args.max_seconds <= 0:
        parser.error("--workers, --max-bytes and --max-seconds must be positive, "
                     "--max-queue at least 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from fast_omniasr import OmniASR

    # Listen first so /health and /ready answer while the model loads.
    service = Service(None, max_bytes=args.max_bytes, max_seconds=args.max_seconds,
                      workers=args.workers, max_queue=args.max_queue)
    server = serve(service, args.host, args.port)
    thread = threading.Thread(target=server.serve_forever, name="http")
    thread.start()
    log.info("listening on %s:%d; loading %s", args.host, args.port, args.model)
    stopping = threading.Event()

    def request_stop(signum, frame):
        log.info("signal %d: shutting down after in-flight requests", signum)
        stopping.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        options = {"device": args.device, "revision": args.revision,
                   "local_files_only": args.local_files_only}
        if args.precision:
            options["precision"] = args.precision
        service.model = OmniASR.from_pretrained(args.model, **options)
    except Exception:
        log.exception("could not load the model")
        server.shutdown()
        server.server_close()
        return 1
    if not stopping.is_set():
        service.ready.set()
        log.info("ready")
    stopping.wait()
    stop(server, service)
    thread.join()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
