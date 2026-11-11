import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "export_onnx.py"
SPEC = importlib.util.spec_from_file_location("export_onnx", SCRIPT)
export_onnx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_onnx)


EXPECTED_CTC_MODELS = {
    "omniASR_CTC_300M",
    "omniASR_CTC_1B",
    "omniASR_CTC_3B",
    "omniASR_CTC_7B",
    "omniASR_CTC_300M_v2",
    "omniASR_CTC_1B_v2",
    "omniASR_CTC_3B_v2",
    "omniASR_CTC_7B_v2",
}


def test_all_upstream_ctc_model_cards_are_supported():
    assert set(export_onnx.CTC_MODEL_CARDS) == EXPECTED_CTC_MODELS


@pytest.mark.parametrize("model", sorted(EXPECTED_CTC_MODELS))
def test_every_ctc_model_card_is_accepted(model, tmp_path):
    args = export_onnx.parse_args(["--model", model, "--output", str(tmp_path / "model.onnx")])
    assert args.model == model


def test_non_ctc_model_card_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        export_onnx.parse_args(
            ["--model", "omniASR_LLM_300M", "--output", str(tmp_path / "model.onnx")]
        )


def test_finetuned_checkpoint_is_accepted(tmp_path):
    checkpoint = tmp_path / "step_100" / "model"
    args = export_onnx.parse_args(
        [
            "--model", "omniASR_CTC_1B_v2",
            "--checkpoint", str(checkpoint),
            "--output", str(tmp_path / "model.onnx"),
        ]
    )
    assert args.model == "omniASR_CTC_1B_v2"
    assert args.checkpoint == checkpoint


def test_finetuned_checkpoint_requires_explicit_base_model(tmp_path, capsys):
    with pytest.raises(SystemExit):
        export_onnx.parse_args(
            ["--checkpoint", str(tmp_path / "model.pt"), "--output", str(tmp_path / "model.onnx")]
        )
    assert "--model is required when --checkpoint is used" in capsys.readouterr().err


def test_official_export_has_no_checkpoint_override(tmp_path):
    args = export_onnx.parse_args(["--output", str(tmp_path / "model.onnx")])
    assert args.model == "omniASR_CTC_300M_v2"
    assert args.checkpoint is None


def test_finetuned_custom_checkpoint_uses_base_config_and_custom_loader(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    args = SimpleNamespace(model="omniASR_CTC_1B_v2", checkpoint=checkpoint)
    calls = []

    class Hub:
        def get_model_config(self, model):
            calls.append(("config", model))
            return "base-config"

        def load_custom_model(self, path, config, **kwargs):
            calls.append(("custom", path, config, kwargs))
            return "fine-tuned-model"

    torch = SimpleNamespace(device=lambda value: value, float32="float32")
    result = export_onnx.load_source_model(
        args, torch, lambda *a, **k: None, lambda: Hub()
    )

    assert result == "fine-tuned-model"
    assert calls[0] == ("config", "omniASR_CTC_1B_v2")
    assert calls[1][:3] == ("custom", checkpoint, "base-config")
    assert calls[1][3] == {"device": "cpu", "dtype": "float32", "mmap": True}


def test_missing_finetuned_checkpoint_fails_before_loading(tmp_path):
    checkpoint = tmp_path / "missing.pt"
    args = SimpleNamespace(model="omniASR_CTC_300M", checkpoint=checkpoint)
    torch = SimpleNamespace(device=lambda value: value, float32="float32")

    with pytest.raises(FileNotFoundError, match="missing.pt"):
        export_onnx.load_source_model(args, torch, None, None)


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("omniASR_CTC_300M", "v1"),
        ("omniASR_CTC_7B", "v1"),
        ("omniASR_CTC_300M_v2", "v2"),
        ("omniASR_CTC_7B_v2", "v2"),
    ],
)
def test_tokenizer_family_is_inferred_from_model_generation(model, family):
    assert export_onnx.tokenizer_family(model) == family


def test_official_tokenizer_is_downloaded_validated_and_bundled(tmp_path):
    destination = tmp_path / "export" / "tokenizer.model"
    downloads = []

    def download(url, path):
        downloads.append(url)
        Path(path).write_bytes(b"official tokenizer")

    result = export_onnx.prepare_tokenizer(
        "omniASR_CTC_1B_v2",
        destination,
        10288,
        download=download,
        get_vocab_size=lambda _: 10288,
    )

    assert result == destination
    assert destination.read_bytes() == b"official tokenizer"
    assert downloads == [export_onnx.TOKENIZER_URLS["v2"]]


def test_tokenizer_vocabulary_must_match_ctc_output(tmp_path):
    def download(_, path):
        Path(path).touch()

    with pytest.raises(ValueError, match="42 classes.*12 entries.*not supported"):
        export_onnx.prepare_tokenizer(
            "omniASR_CTC_300M",
            tmp_path / "tokenizer.model",
            42,
            download=download,
            get_vocab_size=lambda _: 12,
        )


def _logits(frames=6, vocabulary=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, frames, vocabulary)).astype(np.float32) * 4


def test_matching_logits_pass_parity():
    expected = _logits()
    report = export_onnx.compare_logits(expected, expected + 1e-5, label="clip")
    assert report["input"] == "clip"
    assert report["frames"] == 6
    assert report["tokens_equal"] is True
    assert report["near_tie_flips"] == 0


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        (lambda e: e[:, :-1], "shape"),
        (lambda e: np.where(np.arange(e.size).reshape(e.shape) == 3, np.nan, e), "NaN"),
        (lambda e: e + 0.01, "logits differ by up to"),
    ],
)
def test_diverging_logits_fail_parity(actual, message):
    expected = _logits()
    with pytest.raises(ValueError, match=message):
        export_onnx.compare_logits(expected, actual(expected))


def test_flipped_token_fails_even_within_loose_logit_tolerance():
    expected = np.zeros((1, 3, 4), np.float32)
    expected[0, :, 2] = 1.0
    actual = expected.copy()
    actual[0, 1, 3] = 1.5
    with pytest.raises(ValueError, match="greedy token differs at frame 1"):
        export_onnx.compare_logits(expected, actual, atol=2.0)


def test_near_tie_flip_is_tolerated():
    expected = np.zeros((1, 3, 4), np.float32)
    expected[0, :, 2] = 1.0
    expected[0, 1, 3] = 1.0 - 1e-4
    actual = expected.copy()
    actual[0, 1, 3] = 1.0 + 1e-4
    report = export_onnx.compare_logits(expected, actual)
    assert report["near_tie_flips"] == 1


def test_validation_runs_every_input_against_its_reference():
    inputs = {"a": np.zeros((1, 400), np.float32), "b": np.zeros((1, 800), np.float32)}
    references = {"a": _logits(1, seed=1), "b": _logits(2, seed=2)}
    calls = []

    def run(path, given):
        calls.append((path, list(given)))
        return dict(references)

    report = export_onnx.validate_export("model.onnx", references, inputs, run=run)
    assert calls == [("model.onnx", ["a", "b"])]
    assert [probe["input"] for probe in report["probes"]] == ["a", "b"]
    assert report["atol"] == export_onnx.LOGIT_ATOL


def test_default_validation_inputs_cover_speech_and_profile_extremes():
    inputs = export_onnx.validation_inputs()
    assert len(export_onnx.VALIDATION_AUDIO) == 3
    lengths = sorted(audio.shape[1] for audio in inputs.values())
    assert lengths[0] == 400 and lengths[-1] == 480000
    assert len(inputs) == 5
    assert all(audio.dtype == np.float32 and audio.shape[0] == 1 for audio in inputs.values())


def _build(model_path, tokenizer_path):
    model_path.write_bytes(b"new graph")
    (model_path.parent / "weights.data").write_bytes(b"new weights")
    tokenizer_path.write_bytes(b"new tokenizer")


def test_validated_export_is_promoted_with_its_report(tmp_path):
    output = tmp_path / "export" / "model.onnx"
    validated = []

    def validate(path):
        validated.append(path.read_bytes())
        assert not output.exists()
        return {"probes": []}

    export_onnx.staged_export(output, _build, validate)

    assert validated == [b"new graph"]
    assert output.read_bytes() == b"new graph"
    assert (output.parent / "weights.data").read_bytes() == b"new weights"
    assert (output.parent / "tokenizer.model").read_bytes() == b"new tokenizer"
    assert json.loads((output.parent / export_onnx.PARITY_REPORT).read_text()) == {"probes": []}
    assert sorted(p.name for p in output.parent.iterdir()) == sorted(
        ["model.onnx", "weights.data", "tokenizer.model", export_onnx.PARITY_REPORT]
    )


@pytest.mark.parametrize("stage", ["build", "validate"])
def test_failed_export_leaves_previous_output_untouched(tmp_path, stage):
    output = tmp_path / "model.onnx"
    output.write_bytes(b"old graph")
    (tmp_path / "tokenizer.model").write_bytes(b"old tokenizer")

    def build(model_path, tokenizer_path):
        _build(model_path, tokenizer_path)
        if stage == "build":
            raise RuntimeError("export failed")

    def validate(path):
        raise ValueError("noise_400: logits differ")

    with pytest.raises((RuntimeError, ValueError)):
        export_onnx.staged_export(output, build, validate)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx", "tokenizer.model"]
    assert output.read_bytes() == b"old graph"
    assert (tmp_path / "tokenizer.model").read_bytes() == b"old tokenizer"


def test_negative_logit_tolerance_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        export_onnx.parse_args(["--output", str(tmp_path / "m.onnx"), "--logit-atol", "-1"])
