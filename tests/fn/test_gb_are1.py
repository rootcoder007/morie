"""Tests for gb_are1 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_are1 import gibbons_are_sign_wilcoxon


def test_gb_are1_basic():
    from scipy import stats
    assert gibbons_are_sign_wilcoxon(stats.norm.pdf)["are"] == pytest.approx(2 / 3, rel=1e-6)


def test_gb_are1_edge():
    from scipy import stats
    # scale-free: rescaled density gives the same ARE
    a = gibbons_are_sign_wilcoxon(stats.norm.pdf)["are"]
    b = gibbons_are_sign_wilcoxon(lambda x: stats.norm.pdf(x, scale=4))["are"]
    assert b == pytest.approx(a, rel=1e-6)
