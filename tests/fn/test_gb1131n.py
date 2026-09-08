"""Tests for gb1131n (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb1131n import gibbons_spearman_asymp


def test_gb1131n_basic():
    out = gibbons_spearman_asymp(0.5, 26)
    assert out["z"] == pytest.approx(2.5)


def test_gb1131n_edge():
    assert gibbons_spearman_asymp(0.5, 8)["large_sample_ok"] is False
    with pytest.raises(ValueError):
        gibbons_spearman_asymp(1.5, 20)
