"""Tests for gb_are3 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_are3 import gibbons_are_dbl_exp


def test_gb_are3_basic():
    out = gibbons_are_dbl_exp()
    assert out["sign_vs_t"] == pytest.approx(2.0)  # sign BEATS t at the Laplace
    assert out["wilcoxon_vs_t"] == pytest.approx(1.5)


def test_gb_are3_edge():
    with pytest.raises(ValueError):
        gibbons_are_dbl_exp("normal")
