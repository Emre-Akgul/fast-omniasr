"""examples/server: a minimal service with limits, a bounded queue and graceful shutdown (#54)."""

import importlib.util
import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("server", ROOT / "examples" / "server" / "server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def wav(seconds: float, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * rate), np.float32), rate, format="WAV")
    return buffer.getvalue()


class Model:
    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Semaphore(0)
        self.running = 0
        self.peak = 0
        self.closed = False
        self.lock = threading.Lock()

    def transcribe_long(self, waveform, rate, convert_audio):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.started.release()
        self.gate.wait(10)
        with self.lock:
            self.running -= 1
        return type("Result", (), {"text": f"{len(waveform)} samples at {rate}"})()

    def close(self):
        self.closed = True


@pytest.fixture
def running(request):
    options = {"max_bytes": 1_000_000, "max_seconds": 10, "workers": 1, "max_queue": 1,
               **getattr(request, "param", {})}
    model = Model()
    service = server.Service(model, **options)
    http = server.serve(service, "127.0.0.1", 0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    service.ready.set()
    url = f"http://127.0.0.1:{http.server_address[1]}"
    yield url, service, model, http
    model.gate.set()
    if service.ready.is_set():
        server.stop(http, service)


def call(url, path, body=None, timeout=10):
    request = urllib.request.Request(url + path, data=body, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read()), response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def test_transcribes_a_wav(running):
    url, *_ = running
    status, body, _ = call(url, "/transcribe", wav(1.5))
    assert (status, body) == (200, {"text": "24000 samples at 16000", "duration": 1.5})


def test_other_sample_rates_are_passed_on_for_conversion(running):
    url, *_ = running
    assert call(url, "/transcribe", wav(1, rate=8000))[1]["text"] == "8000 samples at 8000"


def test_health_and_readiness(running):
    url, service, *_ = running
    assert call(url, "/health")[:2] == (200, {"status": "ok"})
    assert call(url, "/ready")[:2] == (200, {"ready": True})
    service.ready.clear()
    assert call(url, "/ready")[:2] == (503, {"ready": False})
    status, _, headers = call(url, "/transcribe", wav(1))
    assert (status, headers["Retry-After"]) == (503, "1")


def test_a_body_over_the_byte_limit_is_refused_unread(running):
    url, service, *_ = running
    service.max_bytes = 100
    status, body, _ = call(url, "/transcribe", wav(1))
    assert status == 413 and "100 bytes" in body["error"]


def test_audio_over_the_duration_limit_is_refused(running):
    url, *_ = running
    status, body, _ = call(url, "/transcribe", wav(11))
    assert status == 413 and "limit is 10 s" in body["error"]


def test_audio_that_cannot_be_decoded_is_a_400(running):
    url, *_ = running
    status, body, _ = call(url, "/transcribe", b"not audio at all")
    assert status == 400 and "could not decode audio" in body["error"]


def test_unknown_paths_are_404(running):
    url, *_ = running
    assert call(url, "/nope")[0] == 404
    assert call(url, "/nope", b"x")[0] == 404


def test_a_full_queue_is_refused_and_workers_are_bounded(running):
    url, _, model, _ = running
    model.gate.clear()
    results = []
    threads = [threading.Thread(target=lambda: results.append(call(url, "/transcribe", wav(1))))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    assert model.started.acquire(timeout=5)  # one running; the other waits in the queue
    time.sleep(0.2)
    status, body, headers = call(url, "/transcribe", wav(1))
    assert (status, body, headers["Retry-After"]) == (503, {"error": "busy"}, "1")
    model.gate.set()
    for thread in threads:
        thread.join(10)
    assert sorted(status for status, *_ in results) == [200, 200]
    assert model.peak == 1


def test_shutdown_finishes_in_flight_requests_then_closes_the_model(running):
    url, service, model, http = running
    model.gate.clear()
    results = []
    request = threading.Thread(target=lambda: results.append(call(url, "/transcribe", wav(1))))
    request.start()
    assert model.started.acquire(timeout=5)
    stopper = threading.Thread(target=server.stop, args=(http, service))
    stopper.start()
    time.sleep(0.2)
    assert stopper.is_alive() and not model.closed  # waiting for the request
    model.gate.set()
    stopper.join(10)
    request.join(10)
    assert results[0][0] == 200 and model.closed


def test_limiter_counts_workers_and_queue():
    limiter = server.Limiter(workers=1, max_queue=0)
    with limiter, pytest.raises(server.QueueFull):
        limiter.__enter__()
    with limiter:  # released again
        pass


@pytest.mark.parametrize("argv", [["--workers", "0"], ["--max-queue", "-1"],
                                  ["--max-bytes", "0"], ["--max-seconds", "0"]])
def test_invalid_limits_are_usage_errors(argv):
    with pytest.raises(SystemExit) as raised:
        server.main(["--model", "m", *argv])
    assert raised.value.code == 2
