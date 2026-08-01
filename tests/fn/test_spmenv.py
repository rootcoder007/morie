"""spmenv -- Moran's I moments, Schabenberger & Gotway Sec. 1.3.2 + Problem 1.8."""

import numpy as np
import pytest

from morie.fn.spmenv import schabenberger_moran_expectation as menv


def rook(g):
    n = g * g
    w = np.zeros((n, n))
    for i in range(g):
        for j in range(g):
            k = i * g + j
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, b = i + di, j + dj
                if 0 <= a < g and 0 <= b < g:
                    w[k, a * g + b] = 1.0
    return w


W10 = rook(10)


def _z_example17(seed=0):
    """Example 1.7's design: G(mu(x,y), 1) on a 10x10 lattice."""
    rs = np.random.RandomState(seed)
    x, y = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    return (1.4 + 0.1 * x + 0.2 * y + 0.002 * x ** 2).ravel() + rs.standard_normal(100)


def test_weight_sums_match_the_rook_lattice():
    m = menv(_z_example17(), W10)
    assert (m["S0"], m["S1"], m["S2"]) == (360.0, 720.0, 5312.0)


def test_expectation_is_minus_one_over_n_minus_1():
    """p. 22: the same mean under Gaussianity and randomization."""
    m = menv(_z_example17(), W10)
    assert m["expectation"] == pytest.approx(-1 / 99)


def test_example_1_7_gaussian_sd_reproduced():
    """The book prints sqrt(Var[I]) = 0.0731 under Gaussianity."""
    m = menv(_z_example17(), W10)
    assert m["sd_normal"] == pytest.approx(0.0731, abs=5e-5)


def test_randomization_sd_in_the_books_neighbourhood():
    """The book prints 0.0732 for its own draw; ours differs only through b."""
    m = menv(_z_example17(), W10)
    assert m["sd_randomization"] == pytest.approx(0.0732, abs=3e-4)


def test_normality_variance_is_data_free_but_randomization_is_not():
    rs = np.random.RandomState(3)
    a = menv(rs.standard_normal(100), W10)
    b = menv(rs.standard_normal(100) ** 3, W10)
    assert a["sd_normal"] == b["sd_normal"]
    assert a["sd_randomization"] != b["sd_randomization"]


def test_kurtosis_b_is_the_only_data_channel():
    """Problem 1.8's b: heavy tails move the randomization variance."""
    rs = np.random.RandomState(5)
    light = menv(rs.uniform(-1, 1, 100), W10)
    heavy = menv(rs.standard_t(3, 100), W10)
    assert heavy["kurtosis_b"] > light["kurtosis_b"]


def test_i_invariant_to_affine_rescale():
    """Problem 1.7: Z and a*Z + b give the same I."""
    z = _z_example17()
    a = menv(z, W10)
    b = menv(4.2 * z - 7.0, W10)
    assert a["I"] == pytest.approx(b["I"], abs=1e-12)


def test_positive_autocorrelation_raises_i_above_its_mean():
    """p. 22: I > E[I] when connected sites carry similar values."""
    g = 10
    x, _ = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    smooth = np.sin(x / 2.0).ravel()
    m = menv(smooth + 0.05 * np.random.RandomState(1).standard_normal(100), W10)
    assert m["I"] > m["expectation"]
    assert m["z_normal"] > 2.0


def test_geary_expectation_is_one_and_moves_opposite():
    """p. 22: E[c] = 1, and c < 1 where I > E[I]."""
    g = 10
    x, _ = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    smooth = np.sin(x / 2.0).ravel()
    m = menv(smooth, W10)
    assert m["geary_expectation"] == 1.0
    assert m["geary_c"] < 1.0


def test_checkerboard_gives_negative_autocorrelation():
    g = 10
    x, y = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    board = ((x + y) % 2).astype(float).ravel()
    m = menv(board, W10)
    assert m["I"] < m["expectation"]
    assert m["geary_c"] > 1.0


def test_rejects_constant_field():
    with pytest.raises(ValueError):
        menv(np.ones(100), W10)


def test_rejects_nonzero_diagonal():
    w = W10.copy()
    w[0, 0] = 1.0
    with pytest.raises(ValueError):
        menv(_z_example17(), w)


def test_rejects_shape_mismatch_and_tiny_n():
    with pytest.raises(ValueError):
        menv(np.arange(9.0), W10)
    with pytest.raises(ValueError):
        menv(np.arange(3.0), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float))
