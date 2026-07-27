"""Tests for drovrl.dr_did_overlap_trim."""

import numpy as np
import pytest

from morie.fn.drovrl import dr_did_overlap_trim


def test_drovrl_basic():
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(size=n)
    D = (rng.random(n) < 1 / (1 + np.exp(-2.0 * x))).astype(float)
    fe = rng.normal(size=n)
    y_pre = fe + 0.5 * x + rng.normal(scale=0.5, size=n)
    y_post = fe + 1.3 * x + 1.5 * D + rng.normal(scale=0.5, size=n)
    result = dr_did_overlap_trim(y_pre, y_post, D, x, eps=0.1)
    assert result["att"] == pytest.approx(1.5, abs=0.25)  # measured ~1.51
    assert result["n_trimmed"] > 0
    assert result["n_kept"] + result["n_trimmed"] == n


def test_drovrl_edge():
    y = np.ones(10)
    with pytest.raises(ValueError):
        dr_did_overlap_trim(y, y, np.r_[np.ones(5), np.zeros(5)], np.ones(10), eps=0.6)
    with pytest.raises(ValueError):
        dr_did_overlap_trim(y, y[:5], np.ones(10), np.ones(10))  # length mismatch
