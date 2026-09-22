"""Tests for genrgr.calibration_greg."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.genrgr import calibration_greg


def test_genrgr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    totals = np.random.default_rng(42).normal(0, 1)  # shape (p,) = (1,)
    result = calibration_greg(y, x, weights, totals)
    assert isinstance(result, dict)

    # The function returns a RichResult that behaves like a dict.
    # Required keys per the docstring.
    for key in (
        "total",
        "ht_total",
        "correction",
        "B",
        "residual_totals",
        "design_consistent_regardless_of_model",
        "n",
        "p",
        "method",
    ):
        assert key in result, f"missing key {key!r}"

    # Independent recomputation of the GREG formula on the same inputs.
    w = np.asarray(weights, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    Xv = np.atleast_2d(np.asarray(x, dtype=float))
    if Xv.shape[0] != yv.size:
        Xv = Xv.T
    Tv = np.atleast_1d(np.asarray(totals, dtype=float)).ravel()

    ht_expected = float(np.sum(w * yv))
    Tx_hat_expected = (w[:, None] * Xv).sum(axis=0)
    A_expected = (w[:, None] * Xv).T @ Xv
    B_expected = np.linalg.pinv(A_expected) @ ((w[:, None] * Xv).T @ yv)
    corr_expected = float((Tv - Tx_hat_expected) @ B_expected)
    total_expected = ht_expected + corr_expected

    assert np.isclose(result["ht_total"], ht_expected)
    assert np.isclose(result["correction"], corr_expected)
    assert np.isclose(result["total"], total_expected)
    assert result["design_consistent_regardless_of_model"] is True
    assert result["n"] == 100
    assert result["p"] == 1


def test_genrgr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    totals = np.random.default_rng(42).normal(0, 1)  # shape (p,) = (1,)
    result = calibration_greg(y, x, weights, totals)
    assert isinstance(result, dict)
    assert "total" in result
