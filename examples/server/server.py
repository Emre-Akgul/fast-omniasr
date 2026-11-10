"""A minimal fast-omniasr HTTP service: standard library only, one model loaded at start-up.

    python examples/server/server.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --port 8000
    curl --data-binary @speech.wav http://localhost:8000/transcribe

Endpoints:
- `GET /health`: 200 while the process is up (liveness).
- `GET /ready`: 200 once the model is loaded, 503 before that and while shutting down.
- `POST /transcribe`: the request body is an audio file (WAV, FLAC, ...). Responds with
  `{"text": ..., "duration": seconds}`. 413 when the body or the audio is over the limit,
  400 for audio that can't be decoded, 411 without a Content-Length, 415 for a sample rate
  other than 16 kHz when the `resample` extra isn't installed, 503 with `Retry-After` when
  the queue is full.

`--workers` requests are decoded and transcribed at once (one model instance is shared; the
ONNX Runtime session is thread-safe), at most `--max-queue` more wait, and the rest are
refused right away instead of piling up. Every error response closes its connection.
SIGTERM or SIGINT stops accepting connections, finishes the requests in flight, then closes
the model; idle keep-alive connections don't hold shutdown up.
"""

import argparse
import io
import json
import logging
import signal
import sys
import threading
import time
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


class BadRequest(Exception):
    pass


class TooLarge(Exception):
    pass


class InFlight:
    """Counts requests between reading the body and sending the reply, so shutdown can wait
    for exactly those and not for idle connections."""

    def __init__(self):
        self.count = 0
        self.changed = threading.Condition()

    def __enter__(self):
        with self.changed:
            self.count += 1

    def __exit__(self, *exc_info):
        with self.changed:
            self.count -= 1
            self.changed.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: self.count == 0, timeout)


class Service:
    def __init__(self, model, *, max_bytes: int, max_seconds: float, workers: int,
                 max_queue: int):
        self.model = model
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        self.limiter = Limiter(workers, max_queue)
        self.ready = threading.Event()
        self.in_flight = InFlight()

    def transcribe(self, body: bytes) -> dict:
        import soundfile as sf

        # Check the duration from the header first: a small compressed file can decode to
        # gigabytes of samples.
        try:
            info = sf.info(io.BytesIO(body))
        except Exception as exc:  # libsndfile raises several types
            raise BadRequest(f"could not decode audio: {exc}") from exc
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        if duration > self.max_seconds:
            raise TooLarge(f"audio is {duration:.1f} s; the limit is {self.max_seconds:g} s")
        # Decode inside the limiter, so at most `workers` decoded clips are in memory.
        with self.limiter:
            try:
                waveform, rate = sf.read(io.BytesIO(body), dtype="float32")
            except Exception as exc:  # libsndfile raises several types
                raise BadRequest(f"could not decode audio: {exc}") from exc
            result = self.model.transcribe_long(waveform, rate, convert_audio=True)
        return {"text": result.text, "duration": round(len(waveform) / rate, 3)}


def handler_for(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30  # seconds a connection may sit idle or stall mid-request

        def _reply(self, status: int, body: dict, headers: dict | None = None) -> None:
            # Close on every error and once shutdown has begun: an error may leave the
            # request body unread, and it must not be parsed as the next request.
            close = status >= 400 or not service.ready.is_set()
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
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
            # Count the request before checking readiness: stop() clears readiness and then
            # waits for the count to reach zero, so a request either sees "not ready" or is
            # waited for, never runs on a closed model.
            with service.in_flight:
                self._transcribe()

        def _transcribe(self):
            if not service.ready.is_set():
                self._reply(503, {"error": "not ready"}, {"Retry-After": "1"})
                return
            if "Transfer-Encoding" in self.headers:
                self._reply(411, {"error": "send the body with a Content-Length"})
                return
            value = self.headers.get("Content-Length")
            if value is None:
                self._reply(411, {"error": "Content-Length is required"})
                return
            if not value.strip().isdigit():
                self._reply(400, {"error": f"invalid Content-Length {value!r}"})
                return
            length = int(value)
            if length > service.max_bytes:
                self._reply(413, {"error": f"body is over {service.max_bytes} bytes"})
                return
            body = self.rfile.read(length)
            if len(body) < length:
                self.close_connection = True  # the client went away mid-body
                return
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
    # Connection threads don't block exit: stop() waits for in-flight requests itself, so
    # an idle keep-alive connection can't hold shutdown up.
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):  # the client hung up
            log.debug("%s disconnected", client_address[0])
        else:
            super().handle_error(request, client_address)


def serve(service: Service, host: str, port: int) -> Server:
    return Server((host, port), handler_for(service))


def stop(server: Server, service: Service) -> None:
    """Refuse new work, stop accepting, wait for in-flight requests, close the model."""
    service.ready.clear()
    server.shutdown()  # returns once serve_forever() has stopped accepting
    service.in_flight.wait_idle()
    server.server_close()
    service.model.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", help="fp32 (default), fp16 or bf16")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", help="model cache (default: ~/.cache/fast-omniasr/hub)")
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
    stopping = []  # a plain flag: the handler only appends, nothing in it takes a lock

    def request_stop(signum, frame):
        stopping.append(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        options = {"device": args.device, "revision": args.revision,
                   "cache_dir": args.cache_dir, "local_files_only": args.local_files_only}
        if args.precision:
            options["precision"] = args.precision
        service.model = OmniASR.from_pretrained(args.model, **options)
    except Exception:
        log.exception("could not load the model")
        server.shutdown()
        server.server_close()
        return 1
    if not stopping:
        service.ready.set()
        log.info("ready")
    while not stopping:
        time.sleep(0.2)
    log.info("signal %d: shutting down after in-flight requests", stopping[0])
    stop(server, service)
    thread.join()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
