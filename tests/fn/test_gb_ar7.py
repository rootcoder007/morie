"""Tests for gb_ar7 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_ar7 import gibbons_are_logistic


def test_gb_ar7_basic():
    assert gibbons_are_logistic()["wilcoxon_vs_t"] == pytest.approx(np.pi**2 / 9)


def test_gb_ar7_edge():
    with pytest.raises(ValueError):
        gibbons_are_logistic("normal")
