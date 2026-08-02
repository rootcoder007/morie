"""Tests for sptau.spatial_autocorrelation (Moran's I)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.sptau import spatial_autocorrelation


def _path_w(n):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def _rook_w(side):
    n = side * side
    W = np.zeros((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r + 1 < side:
                W[i, i + side] = W[i + side, i] = 1.0
            if c + 1 < side:
                W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def test_sptau_matches_a_hand_computed_value():
    """Path graph with x = (1, 2, 4): z = (-4/3, -1/3, 5/3),
    sum_ij w_ij z_i z_j = 2(z0 z1 + z1 z2) = -2/9, sum z^2 = 42/9, S0 = 4,
    so I = (3/4)(-2/42) = -1/28 (Schabenberger & Gotway 2005, eq. 1.14)."""
    r = spatial_autocorrelation([1.0, 2.0, 4.0], _path_w(3))
    assert float(r["statistic"]) == pytest.approx(-1.0 / 28.0, abs=1e-12)


def test_sptau_expectation_matches_the_book_lattice():
    """10 x 10 rook lattice: E[I] = -1/99 = -0.0101, the value printed in
    Schabenberger & Gotway (2005), Example 1.7, p. 22."""
    rng = np.random.default_rng(0)
    r = spatial_autocorrelation(rng.normal(size=100), _rook_w(10))
    assert float(r["expectation"]) == pytest.approx(-1.0 / 99.0, abs=1e-12)
    assert float(r["expectation"]) == pytest.approx(-0.0101, abs=5e-5)


def test_sptau_detects_clustering_and_dispersion():
    side = 8
    W = _rook_w(side)
    grad = np.add.outer(np.arange(side), np.arange(side)).ravel().astype(float)
    pos = spatial_autocorrelation(grad, W)
    assert float(pos["statistic"]) > 0.5
    assert float(pos["p_value"]) < 1e-6

    checker = np.indices((side, side)).sum(axis=0).ravel() % 2 * 2.0 - 1.0
    neg = spatial_autocorrelation(checker, W)
    assert float(neg["statistic"]) < float(neg["expectation"])
    assert float(neg["p_value"]) < 1e-6


def test_sptau_null_rejection_rate_is_nominal():
    """White noise on the lattice: measured 0/20 rejections at alpha = 0.01
    across seeds 0..19."""
    W = _rook_w(6)
    rej = 0
    for s in range(20):
        rng = np.random.default_rng(s)
        r = spatial_autocorrelation(rng.normal(size=36), W)
        rej += float(r["p_value"]) < 0.01
    assert rej <= 2


def test_sptau_rejects_mismatched_w():
    with pytest.raises(ValueError, match="square matrix"):
        spatial_autocorrelation([1.0, 2.0, 3.0], np.zeros((2, 2)))
