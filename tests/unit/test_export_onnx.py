import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
