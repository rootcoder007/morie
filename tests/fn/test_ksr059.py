"""Tests for ksr059 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr059 import kosorok_ch2_kmt_strong_approximation


def test_ksr059_basic():
    out = kosorok_ch2_kmt_strong_approximation(1000, x=2.0, a=1.0, b=1.0, c=1.0)
    assert out["probability_bound"] == pytest.approx(np.exp(-2.0))


def test_ksr059_edge():
    # the universal constants are not stated in the literature, so the
    # module refuses to invent them
    with pytest.raises(ValueError, match="universal"):
        kosorok_ch2_kmt_strong_approximation(1000, x=1.0)
