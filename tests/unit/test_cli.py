"""The fast-omniasr command (#51)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fast_omniasr import OmniASR, Transcription, Word, hub
from fast_omniasr.cli import main
from fast_omniasr.llm import LLMTranscription, OmniASRLLM


class Model:
    device = "cpu"
    snapshot = hub.Snapshot("owner/repo", "main", "abc", Path("/cache/bundle"))
    backend = SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    tokenizer = SimpleNamespace(vocabulary_size=7)

    def __init__(self):
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True

    def transcribe(self, path, **options):
        self.calls.append(("transcribe", path, options))
        if "missing" in path:
            raise OSError("no such file")
        words = (Word(0.0, 0.5, "hello"),) if options.get("word_timestamps") else ()
        return Transcription(f"text of {path}", (1, 2), (1, 2, 3), words=words)

    def transcribe_long(self, path, **options):
        self.calls.append(("transcribe_long", path, options))
        return Transcription("long", (1,), (1, 2, 3))


class LLM(Model):
    def transcribe(self, path, **options):
        self.calls.append(("transcribe", path, options))
        return LLMTranscription("words", [1], "eos")


@pytest.fixture
def ctc(monkeypatch):
    model, loads = Model(), []

    def load(repo, **options):
        loads.append((repo, options))
        return model

    monkeypatch.setattr(OmniASR, "from_pretrained", load)
    model.loads = loads
    return model


@pytest.fixture
def llm(monkeypatch):
    model, loads = LLM(), []

    def load(repo, **options):
        loads.append((repo, options))
        return model

    monkeypatch.setattr(OmniASRLLM, "from_pretrained", load)
    model.loads = loads
    return model


def run(*argv):
    return main(list(argv))


def lines(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_text_output_is_the_transcript(ctc, capsys):
    assert run("transcribe", "a.wav", "--model", "owner/repo") == 0
    assert capsys.readouterr().out == "text of a.wav\n"
    assert ctc.closed


def test_several_files_are_labelled(ctc, capsys):
    assert run("transcribe", "a.wav", "b.wav", "--model", "m") == 0
    assert capsys.readouterr().out == "a.wav: text of a.wav\nb.wav: text of b.wav\n"


def test_load_options_reach_the_model(ctc):
    run("transcribe", "a.wav", "--model", "m", "--revision", "abc", "--cache-dir", "c",
        "--local-files-only", "--device", "cuda", "--precision", "fp16", "--threads", "2",
        "--backend", "tensorrt")
    assert ctc.loads == [("m", {"revision": "abc", "cache_dir": "c", "local_files_only": True,
                                "device": "cuda", "precision": "fp16", "threads": 2,
                                "backend": "tensorrt"})]


def test_unset_options_keep_the_library_defaults(ctc):
    run("transcribe", "a.wav", "--model", "m")
    assert ctc.loads == [("m", {"revision": None, "cache_dir": None, "local_files_only": False,
                                "backend": "onnx"})]
    assert ctc.calls == [("transcribe", "a.wav", {"convert_audio": False,
                                                  "word_timestamps": False})]


def test_json_has_the_result_fields_and_path(ctc, capsys):
    assert run("transcribe", "a.wav", "--model", "m", "--json") == 0
    assert lines(capsys) == [{"path": "a.wav", "text": "text of a.wav", "token_ids": [1, 2],
                              "logits_shape": [1, 2, 3], "chunks": [], "words": [],
                              "token_times": []}]


def test_json_keeps_non_ascii_text(ctc, capsys, monkeypatch):
    monkeypatch.setattr(Model, "transcribe",
                        lambda self, path, **options: Transcription("şüphe", (), ()))
    run("transcribe", "a.wav", "--model", "m", "--json")
    assert "şüphe" in capsys.readouterr().out


def test_word_timestamps(ctc, capsys):
    assert run("transcribe", "a.wav", "--model", "m", "--word-timestamps") == 0
    assert capsys.readouterr().out == "text of a.wav\n0.00\t0.50\thello\n"


def test_word_timestamps_in_json(ctc, capsys):
    run("transcribe", "a.wav", "--model", "m", "--word-timestamps", "--json")
    assert lines(capsys)[0]["words"] == [{"start": 0.0, "end": 0.5, "word": "hello"}]


@pytest.mark.parametrize("flags, options", [
    (["--long"], {"vad_filter": False}),
    (["--vad"], {"vad_filter": True}),
    (["--chunk-length", "10"], {"vad_filter": False, "chunk_length": 10.0}),
    (["--vad", "--chunk-length", "5"], {"vad_filter": True, "chunk_length": 5.0}),
])
def test_long_audio_options(ctc, flags, options):
    assert run("transcribe", "a.wav", "--model", "m", *flags) == 0
    assert ctc.calls == [("transcribe_long", "a.wav",
                          {"convert_audio": False, "word_timestamps": False, **options})]


def test_a_failed_file_does_not_stop_the_rest(ctc, capsys):
    assert run("transcribe", "missing.wav", "b.wav", "--model", "m") == 1
    captured = capsys.readouterr()
    assert "error: missing.wav: no such file" in captured.err
    assert captured.out == "b.wav: text of b.wav\n"


def test_a_failed_file_is_reported_in_json(ctc, capsys):
    assert run("transcribe", "missing.wav", "b.wav", "--model", "m", "--json") == 1
    results = lines(capsys)
    assert results[0] == {"path": "missing.wav", "error": "no such file"}
    assert results[1]["path"] == "b.wav"


def test_a_model_that_will_not_load_is_a_message(monkeypatch, capsys):
    def load(repo, **options):
        raise hub.LocalBundleNotFoundError("not cached")

    monkeypatch.setattr(OmniASR, "from_pretrained", load)
    assert run("transcribe", "a.wav", "--model", "m") == 1
    assert capsys.readouterr().err == "error: not cached\n"


def test_llm_options_reach_the_model(llm, capsys):
    assert run("transcribe", "a.wav", "--model", "m", "--kind", "llm", "--lang", "eng_Latn",
               "--max-new-tokens", "9", "--repetition-penalty", "1.1", "--low-memory") == 0
    assert capsys.readouterr().out == "words\n"
    assert llm.loads == [("m", {"revision": None, "cache_dir": None, "local_files_only": False,
                                "low_memory": True})]
    assert llm.calls == [("transcribe", "a.wav", {"convert_audio": False, "lang": "eng_Latn",
                                                  "max_new_tokens": 9,
                                                  "repetition_penalty": 1.1})]


def test_llm_json_has_the_stop_reason(llm, capsys):
    run("transcribe", "a.wav", "--model", "m", "--kind", "llm", "--json")
    assert lines(capsys)[0]["stop_reason"] == "eos"


@pytest.mark.parametrize("flags", [["--lang", "eng_Latn"], ["--max-new-tokens", "0"],
                                   ["--repetition-penalty", "1"], ["--low-memory"]])
def test_llm_options_are_refused_on_ctc(flags, capsys):
    with pytest.raises(SystemExit) as stopped:
        run("transcribe", "a.wav", "--model", "m", *flags)
    assert stopped.value.code == 2 and "doesn't apply to --kind ctc" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--backend", "onnx"], ["--word-timestamps"]])
def test_ctc_options_are_refused_on_llm(flags, capsys):
    with pytest.raises(SystemExit) as stopped:
        run("transcribe", "a.wav", "--model", "m", "--kind", "llm", *flags)
    assert stopped.value.code == 2 and "doesn't apply to --kind llm" in capsys.readouterr().err


def test_inspect_reports_the_model(ctc, capsys):
    assert run("inspect", "--model", "m", "--json") == 0
    assert lines(capsys) == [{"kind": "ctc", "repo_id": "owner/repo", "revision": "main",
                              "commit": "abc", "path": str(Path("/cache/bundle")),
                              "backend": "onnx", "vocabulary_size": 7,
                              "providers": ["CPUExecutionProvider"]}]


def test_inspect_reports_an_llm(llm, capsys):
    assert run("inspect", "--model", "m", "--kind", "llm", "--precision", "fp16") == 0
    output = capsys.readouterr().out
    assert "kind: llm" in output and "device: cpu" in output and "precision: fp16" in output


def test_languages_are_listed(monkeypatch, capsys, tmp_path):
    table = tmp_path / "languages.json"
    table.write_text('{"tur_Latn": 1, "eng_Latn": 0}', encoding="utf-8")
    seen = {}

    def download(repo, required, **options):
        seen.update(options, required=required({"languages": {"mapping": "languages.json"}}))
        return {}, {"languages.json": table}

    monkeypatch.setattr(hub, "download_verified", download)
    assert run("languages", "--model", "m") == 0
    assert capsys.readouterr().out == "eng_Latn\ntur_Latn\n"
    assert seen["required"] == {"languages.json"} and seen["dependencies"] is True
    assert run("languages", "--model", "m", "--json") == 0
    assert lines(capsys) == [["eng_Latn", "tur_Latn"]]


def test_a_model_without_languages_is_a_message(monkeypatch, capsys):
    monkeypatch.setattr(hub, "download_verified",
                        lambda repo, required, **options: required({}))
    assert run("languages", "--model", "owner/ctc") == 1
    assert capsys.readouterr().err == "error: owner/ctc has no language table\n"


def test_diagnose_reports_the_environment(capsys, tmp_path):
    assert run("diagnose", "--json", "--cache-dir", str(tmp_path)) == 0
    info = lines(capsys)[0]
    assert info["python"].count(".") == 2
    assert info["packages"]["numpy"]
    assert info["cache_dir"] == str(tmp_path) and info["cached_models"] == []
    assert isinstance(info["tensorrt_available"], bool)


def test_diagnose_text_says_what_is_missing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("fast_omniasr.cli.PACKAGES", ("no-such-package",))
    assert run("diagnose", "--cache-dir", str(tmp_path)) == 0
    assert "no-such-package: not installed" in capsys.readouterr().out


def test_a_broken_onnxruntime_is_reported_not_raised(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises ImportError
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    assert run("diagnose", "--json", "--cache-dir", str(tmp_path)) == 0
    info = lines(capsys)[0]
    assert info["onnxruntime_providers"] is None and info["tensorrt_available"] is False


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as stopped:
        run()
    assert stopped.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as stopped:
        run("--version")
    assert stopped.value.code == 0 and "fast-omniasr" in capsys.readouterr().out


def test_module_entry_point_runs_without_heavy_imports():
    code = ("import sys; from fast_omniasr.cli import main; main(['diagnose', '--json']); "
            "assert 'torch' not in sys.modules and 'huggingface_hub' not in sys.modules")
    src = Path(__file__).resolve().parents[2] / "src"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(src)})


def test_python_dash_m_works():
    src = Path(__file__).resolve().parents[2] / "src"
    done = subprocess.run([sys.executable, "-m", "fast_omniasr", "--help"], check=True,
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(src)})
    assert "transcribe" in done.stdout


def test_a_closed_pipe_is_not_an_error(ctc, monkeypatch):
    def broken(*args, **options):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", broken)
    monkeypatch.setattr("fast_omniasr.cli.os.dup2", lambda *args: None)
    monkeypatch.setattr("sys.stdout.fileno", lambda: 1, raising=False)
    assert run("transcribe", "a.wav", "--model", "m") == 0
