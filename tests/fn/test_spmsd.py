"""Tests for spmsd.schabenberger_mean_square_diff (Stein Ch. 2.6)."""

from morie.fn import _array_core as np
from morie.fn.spmsd import schabenberger_mean_square_diff


def test_spmsd_gaussian_covariance_is_differentiable():
    # C(h) = exp(-h^2): C''(0) = -2 exactly; derivative field cov = +2
    r = schabenberger_mean_square_diff(
        lambda h: np.exp(-(np.asarray(h) ** 2)), m=1)
    assert r["converged"] is True
    assert abs(r["derivative_2m"] + 2.0) < 1e-6
    assert abs(r["derivative_cov"] - 2.0) < 1e-6


def test_spmsd_exponential_kink_is_not_differentiable():
    # C(h) = exp(-|h|) has a kink at 0: the central difference diverges,
    # which IS the diagnostic
    r = schabenberger_mean_square_diff(
        lambda h: np.exp(-abs(np.asarray(h))), m=1)
    assert r["converged"] is False
    assert abs(r["derivative_2m"]) > 1e3
