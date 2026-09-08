import numpy as np
import pytest

from fast_omniasr.audio import prepare_audio


def test_normalization_and_input_unchanged():
    source = np.tile(np.array([1, 3], dtype=np.float32), 400)
    original = source.copy()
    result = prepare_audio(source)
    assert result.shape == (1, 800)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(source, original)
    assert abs(result.mean()) < 1e-7
    assert result.var() == pytest.approx(1 / (1 + 1e-5), abs=1e-6)


def test_silence_is_finite():
    np.testing.assert_array_equal(prepare_audio(np.ones(16000)), np.zeros((1,16000)))


@pytest.mark.parametrize("wave,rate", [(np.zeros(399),16000), (np.zeros((800,2)),16000),
                                      (np.zeros(800),8000), (np.full(800,np.nan),16000)])
def test_invalid_inputs(wave, rate):
    with pytest.raises(ValueError):
        prepare_audio(wave, rate)
