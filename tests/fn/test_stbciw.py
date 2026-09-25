"""Tests for stbciw.stabilized_censoring_weights (Robins 1993)."""

import math

import pytest

from morie.fn.stbciw import stabilized_censoring_weights


def test_stbciw_basic():
    """w_i = prod_t P(C_t = 0 | C_{t-1} = 0) / prod_t P(C_t = 0 | history),
    the unstabilised weight 1 / prod_t P(C_t = 0 | history)."""
    C = [[0.9, 0.8, 0.95], [0.7, 0.9, 0.85], [0.99, 0.97, 0.9]]
    N = [[0.85, 0.88, 0.9]] * 3
    r = stabilized_censoring_weights(C, numerator=N)
    w = [math.prod(N[i]) / math.prod(C[i]) for i in range(3)]
    assert r["weights"] == pytest.approx(w, rel=1e-15)
    assert r["unstabilized"] == pytest.approx([1 / math.prod(c) for c in C], rel=1e-15)
    assert r["mean_weight"] == pytest.approx(sum(w) / 3, rel=1e-15)
    assert r["estimate"] == r["mean_weight"]
    assert r["max_weight"] == max(r["weights"])
    assert (r["n"], r["n_times"]) == (3, 3)


def test_stbciw_edge():
    """Without a numerator the weights are unstabilised; probabilities
    outside (0, 1], ragged rows and mismatched shapes raise."""
    C = [[0.5, 0.5], [0.25, 1.0]]
    r = stabilized_censoring_weights(C)
    assert r["weights"] == [4.0, 4.0]
    with pytest.raises(ValueError):
        stabilized_censoring_weights([[0.0, 0.5]])
    with pytest.raises(ValueError):
        stabilized_censoring_weights([[0.5, 0.5], [0.5]])
    with pytest.raises(ValueError):
        stabilized_censoring_weights(C, numerator=[[1.0, 1.0]])
    with pytest.raises(ValueError):
        stabilized_censoring_weights(C, H=[[1.0]])
