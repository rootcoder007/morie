"""Tests for ksr053 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr053 import kosorok_ch2_kaplan_meier_inverse


def test_ksr053_basic():
    out = kosorok_ch2_kaplan_meier_inverse(lambda u: np.exp(-0.5 * u),
                                           lambda u: np.exp(-0.3 * u), None,
                                           lambda u: u, 1.0)
    assert np.isfinite(out["inverse"])


def test_ksr053_edge():
    # a hazard-like L vanishing at 0 makes the integrand undefined
    with pytest.raises(ValueError):
        kosorok_ch2_kaplan_meier_inverse(lambda u: np.exp(-0.5 * u),
                                         lambda u: 0.5 * u, None, lambda u: u, 1.0)
