"""Test white_noise_gen (whtns)."""

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.whtns import white_noise_gen, whtns


class TestWhiteNoise:
    def test_length(self):
        result = white_noise_gen(100, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 100

    def test_reproducible(self):
        a = white_noise_gen(50, seed=7)
        b = white_noise_gen(50, seed=7)
        assert np.allclose(a.value, b.value)

    def test_alias(self):
        assert whtns is white_noise_gen
