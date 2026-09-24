"""Tests for km130.kamath_ch9_input_alignment_loss."""

import math

from morie.fn import _array_core as np

from morie.fn.km130 import kamath_ch9_input_alignment_loss


def _mse(prediction, target):
    return float(np.mean(
        (np.array(prediction, dtype=float)
         - np.array(target, dtype=float)) ** 2
    ))


def _add(P, F):
    return np.array(P, dtype=float) + np.array(F, dtype=float)


def test_km130_basic():
    """Test basic functionality with a 3-D stack of candidates."""
    P_X = [[[0.0]], [[1.0]]]
    F_T = [[1.0]]
    t = [[1.0]]
    result = kamath_ch9_input_alignment_loss(
        P_X, F_T, t, llm=_add, loss_fn=_mse
    )
    assert isinstance(result, dict)
    for key in ("estimate", "argmin", "losses", "n_candidates", "n", "method"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert result["argmin"] == 0
    assert result["n_candidates"] == 2


def test_km130_edge():
    """Test edge case with a 2-D single prompt-feature matrix."""
    P_X = [[1.0]]
    F_T = [[1.0]]
    t = [[1.0]]
    result = kamath_ch9_input_alignment_loss(
        P_X, F_T, t, llm=_add, loss_fn=_mse
    )
    assert isinstance(result, dict)
    for key in ("estimate", "argmin", "losses", "n_candidates", "n", "method"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert result["n_candidates"] == 1
    assert result["argmin"] == 0
