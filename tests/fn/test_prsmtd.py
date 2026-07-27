"""Tests for prsmtd.propensity_score_method."""

import numpy as np
import pytest

from morie.fn.prsmtd import propensity_score_method


def test_prsmtd_basic():
    rng = np.random.default_rng(42)
    n, T = 300, 3
    H = rng.normal(size=(n, T))
    A = np.zeros((n, T))
    ever = np.zeros(n, dtype=bool)
    for t in range(T):
        start = (rng.random(n) < 0.25) & ~ever
        A[start, t] = 1
        ever |= start
    result = propensity_score_method(A, H)
    m = result["matched_idx"]
    assert m.shape[0] > 0
    for t, i, j in m:
        assert A[i, t] == 1 and A[i, :t].sum() == 0  # newly treated at t
        assert A[j, : t + 1].sum() == 0  # control untreated through t
    # without replacement: no control reused
    assert len({j for _, _, j in m}) == m.shape[0]


def test_prsmtd_edge():
    with pytest.raises(ValueError):
        propensity_score_method([[1, 0]], [[0.5]])  # A/H shape mismatch
    with pytest.raises(ValueError):
        propensity_score_method([[0.5]], [[0.5]])  # non-binary A
