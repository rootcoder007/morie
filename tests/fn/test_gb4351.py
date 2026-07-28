"""Tests for gb4351 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb4351 import gibbons_ks_chi2_approx


def test_gb4351_basic():
    out = gibbons_ks_chi2_approx(400, 0.05)
    assert out["p_value"] == pytest.approx(np.exp(-2 * 400 * 0.0025), abs=1e-12)


def test_gb4351_edge():
    with pytest.raises(ValueError):
        gibbons_ks_chi2_approx(400, 0.0)
