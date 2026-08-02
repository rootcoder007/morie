"""Tests for gb_are4 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_are4 import gibbons_are_kw


def test_gb_are4_basic():
    for d in ("uniform", "normal", "logistic", "double_exponential"):
        assert gibbons_are_kw(d)["above_bound"] is True


def test_gb_are4_edge():
    with pytest.raises(ValueError):
        gibbons_are_kw("weibull")
