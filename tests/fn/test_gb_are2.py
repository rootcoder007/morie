"""Tests for gb_are2 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_are2 import gibbons_are_normal_case


def test_gb_are2_basic():
    out = gibbons_are_normal_case()
    assert out["wilcoxon_vs_t"] == pytest.approx(3 / np.pi)
    assert out["sign_vs_t"] == pytest.approx(2 / np.pi)


def test_gb_are2_edge():
    with pytest.raises(ValueError):
        gibbons_are_normal_case("weibull")
