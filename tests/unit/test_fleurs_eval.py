"""tools/fleurs_eval.py: manifest selection, verified extraction, and regression gates."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fleurs_eval", ROOT / "tools" / "fleurs_eval.py")
fleurs_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleurs_eval)


def _tsv(path, rows):
    path.write_text("".join(
        f"{sid}\t{name}\t{ref}\t{ref.lower()}\tp h o\t{samples}\tMALE\n"
        for sid, name, ref, samples in rows), encoding="utf-8")


def test_read_tsv_parses_fleurs_rows(tmp_path):
    _tsv(tmp_path / "test.tsv", [("1", "a.wav", "Hello, world.", 32000)])
    assert fleurs_eval.read_tsv(tmp_path / "test.tsv") == [
        {"id": "1", "file_name": "a.wav", "reference": "Hello, world.", "num_samples": 32000}]


def test_read_tsv_rejects_malformed_rows(tmp_path):
    (tmp_path / "bad.tsv").write_text("1\ta.wav\tonly three\n")
    with pytest.raises(ValueError, match="7 tab-separated"):
        fleurs_eval.read_tsv(tmp_path / "bad.tsv")


def test_selection_is_deterministic_short_and_one_per_sentence():
    rows = [{"id": str(i // 2), "file_name": f"{i}.wav", "reference": f"text {i}",
             "num_samples": 16000 * (5 if i != 3 else 40)} for i in range(40)]
    first = fleurs_eval.select_utterances(rows, 10, 20 * 16000)
    again = fleurs_eval.select_utterances(list(reversed(rows)), 10, 20 * 16000)
    assert first == again  # independent of TSV order
    assert len({r["id"] for r in first}) == 10
    assert all(r["num_samples"] <= 20 * 16000 for r in first)
    with pytest.raises(ValueError, match="usable"):
        fleurs_eval.select_utterances(rows, 30, 20 * 16000)


def _archive(path, files):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"test/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _utterance(name, data):
    return {"file_name": name, "reference": "x", "sha256": hashlib.sha256(data).hexdigest()}


def test_extract_members_verifies_and_extracts_only_listed(tmp_path):
    _archive(tmp_path / "test.tar.gz", {"a.wav": b"aaa", "b.wav": b"bbb", "c.wav": b"ccc"})
    fleurs_eval.extract_members(tmp_path / "test.tar.gz", tmp_path / "out",
                                [_utterance("a.wav", b"aaa"), _utterance("c.wav", b"ccc")])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.wav", "c.wav"]


def test_extract_members_rejects_mismatch_and_missing(tmp_path):
    _archive(tmp_path / "test.tar.gz", {"a.wav": b"aaa"})
    with pytest.raises(RuntimeError, match="does not match"):
        fleurs_eval.extract_members(tmp_path / "test.tar.gz", tmp_path / "o1",
                                    [_utterance("a.wav", b"other")])
    assert not (tmp_path / "o1" / "a.wav").exists()
    with pytest.raises(RuntimeError, match="missing"):
        fleurs_eval.extract_members(tmp_path / "test.tar.gz", tmp_path / "o2",
                                    [_utterance("z.wav", b"zzz")])


def test_fetch_audio_skips_download_when_cached(tmp_path, monkeypatch):
    data = b"cached"
    manifest = {"revision": "f" * 40, "languages": {"en_us": {
        "utterances": [_utterance("a.wav", data)]}}}
    (tmp_path / "en_us").mkdir()
    (tmp_path / "en_us" / "a.wav").write_bytes(data)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)  # must not import
    paths = fleurs_eval.fetch_audio(manifest, tmp_path, ["en_us"], None)
    assert paths == {"en_us": [tmp_path / "en_us" / "a.wav"]}


MANIFEST = {"languages": {"en_us": {"metric": "wer"}, "cmn_hans_cn": {"metric": "cer"}}}


@pytest.mark.parametrize("results, failures", [
    ({"en_us": {"wer": 10.0, "cer": 4.0}}, []),
    ({"en_us": {"wer": 12.5, "cer": 4.0}}, ["en_us: WER 12.50 > 12.00"]),
    ({"en_us": {"wer": 10.0, "cer": 6.0}}, ["en_us: CER 6.00 > 5.00"]),
    ({"cmn_hans_cn": {"wer": 99.0, "cer": 8.0}}, []),  # Mandarin: WER not gated
    ({"cmn_hans_cn": {"wer": 0.0, "cer": 9.5}}, ["cmn_hans_cn: CER 9.50 > 9.00"]),
])
def test_gate(results, failures):
    thresholds = {"languages": {"en_us": {"max_wer": 12.0, "max_cer": 5.0},
                                "cmn_hans_cn": {"max_wer": None, "max_cer": 9.0}}}
    assert fleurs_eval.check(results, thresholds, MANIFEST) == failures


def test_languages_without_thresholds_are_not_gated():
    assert fleurs_eval.check({"en_us": {"wer": 99.0, "cer": 99.0}}, {"languages": {}},
                             MANIFEST) == []


def test_committed_manifest_and_thresholds_are_consistent():
    manifest = json.loads(fleurs_eval.MANIFEST.read_text())
    fleurs_eval.validate_manifest(manifest)
    assert set(manifest["languages"]) == set(fleurs_eval.LANGUAGES)
    for fleurs, entry in manifest["languages"].items():
        assert (entry["lang"], entry["metric"]) == fleurs_eval.LANGUAGES[fleurs]
        assert all(u["seconds"] <= fleurs_eval.MAX_SECONDS for u in entry["utterances"])
    thresholds = json.loads(fleurs_eval.THRESHOLDS.read_text())
    for name, model in thresholds["models"].items():
        assert len(model["revision"]) == 40 and model["per_language"] >= 1
        assert set(model["languages"]) == set(manifest["languages"]), name
        for fleurs, limits in model["languages"].items():
            assert limits["max_cer"] is not None
            if manifest["languages"][fleurs]["metric"] == "wer":
                assert limits["max_wer"] is not None


@pytest.mark.parametrize("bad", [
    {"dataset": "other", "revision": "f" * 40, "languages": {}},
    {"dataset": "google/fleurs", "revision": "main", "languages": {}},
    {"dataset": "google/fleurs", "revision": "f" * 40,
     "languages": {"en_us": {"metric": "bleu", "utterances": []}}},
    {"dataset": "google/fleurs", "revision": "f" * 40,
     "languages": {"en_us": {"metric": "wer", "utterances": [
         {"file_name": "a", "reference": "x", "sha256": "0" * 64}] * 2}}},
])
def test_validate_manifest_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        fleurs_eval.validate_manifest(bad)


def test_run_gate_exit_codes_and_report(tmp_path, monkeypatch):
    manifest = {"dataset": "google/fleurs", "revision": "f" * 40, "languages": {
        "en_us": {"lang": "eng_Latn", "metric": "wer", "utterances": [
            {"file_name": "a.wav", "reference": "hello world", "sha256": "0" * 64}]}}}
    thresholds = {"models": {"ctc": {"repo": "o/ctc", "revision": "c" * 40, "per_language": 1,
                                     "languages": {"en_us": {"max_wer": 10.0, "max_cer": 5.0}}}}}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "t.json").write_text(json.dumps(thresholds))
    monkeypatch.setattr(fleurs_eval, "fetch_audio",
                        lambda m, d, langs, n, c=None: {"en_us": [tmp_path / "a.wav"]})

    def fake_transcriber(text):
        class Model:
            closed = False

            def close(self):
                self.closed = True

        model = Model()
        return lambda *a: (model, lambda path, lang: (text, None)), model

    args = ["run", "--model", "ctc", "--manifest", str(tmp_path / "m.json"),
            "--thresholds", str(tmp_path / "t.json"), "--gate",
            "--report", str(tmp_path / "r.json")]
    factory, model = fake_transcriber("hello world")
    monkeypatch.setattr(fleurs_eval, "transcriber", factory)
    assert fleurs_eval.main(args) == 0 and model.closed
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["results"]["en_us"]["wer"] == 0 and report["failures"] == []
    assert report["revision"] == "c" * 40 and report["dataset_revision"] == "f" * 40
    factory, _ = fake_transcriber("hello word")
    monkeypatch.setattr(fleurs_eval, "transcriber", factory)
    assert fleurs_eval.main(args) == 1
    assert json.loads((tmp_path / "r.json").read_text())["failures"] == [
        "en_us: WER 50.00 > 10.00", "en_us: CER 9.09 > 5.00"]
