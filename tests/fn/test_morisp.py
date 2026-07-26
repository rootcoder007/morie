"""morisp: Moran's I (Schabenberger & Gotway Eq. 1.14 p.21; Moran 1950)."""

import numpy as np
import pytest

from morie.fn.morisp import morans_i


def _rook(nr, nc):
    """Rook (edge-sharing) contiguity on an nr x nc lattice."""
    n = nr * nc
    W = np.zeros((n, n))
    for i in range(nr):
        for j in range(nc):
            a = i * nc + j
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p, q = i + di, j + dj
                if 0 <= p < nr and 0 <= q < nc:
                    W[a, p * nc + q] = 1.0
    return W


def test_morisp_expectation_matches_the_books_worked_example():
    """Example 1.7 (p. 22) uses a 10x10 lattice and prints E[I] = -0.0101.

    E[I] = -1/(n-1) = -1/99 = -0.010101..., which rounds to the printed value.
    """
    rng = np.random.default_rng(0)
    W = _rook(10, 10)
    r = morans_i(rng.standard_normal(100), W)
    assert r["n"] == 100
    assert r["expected"] == pytest.approx(-1.0 / 99.0)
    assert round(r["expected"], 4) == -0.0101


def test_morisp_expectation_is_not_zero():
    """E[I] = -1/(n-1). Comparing I against 0 biases toward 'positive'."""
    W = _rook(3, 4)
    r = morans_i(np.arange(12.0), W)
    assert r["expected"] == pytest.approx(-1.0 / 11.0)


def test_morisp_perfect_positive_pattern_is_strongly_above_expectation():
    """Two solid blocks: neighbours nearly always share a value."""
    x = np.concatenate([np.ones(50), -np.ones(50)])
    W = _rook(10, 10)
    r = morans_i(x, W)
    assert r["estimate"] > 0.8
    assert r["estimate"] > r["expected"]


def test_morisp_checkerboard_is_maximally_negative():
    """Every rook neighbour has the opposite sign, so I = -1 exactly."""
    nr = nc = 10
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    x = ((ii + jj) % 2 * 2.0 - 1.0).ravel()
    r = morans_i(x, _rook(nr, nc))
    assert r["estimate"] == pytest.approx(-1.0)


def test_morisp_random_data_sits_near_the_null():
    """Averaged over permutations, I concentrates on E[I] = -1/(n-1)."""
    rng = np.random.default_rng(101)
    W = _rook(6, 6)
    x = rng.standard_normal(36)
    vals = [morans_i(rng.permutation(x), W)["estimate"] for _ in range(400)]
    assert float(np.mean(vals)) == pytest.approx(-1.0 / 35.0, abs=0.03)


def test_morisp_ignores_self_weights():
    """A site is not its own neighbour; a nonzero diagonal must not change I."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(16)
    W = _rook(4, 4)
    W_diag = W + np.eye(16) * 3.0
    assert morans_i(x, W)["estimate"] == pytest.approx(morans_i(x, W_diag)["estimate"])


def test_morisp_is_invariant_to_affine_rescaling_of_x():
    """I is built from deviations over a sum of squared deviations."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(25)
    W = _rook(5, 5)
    assert morans_i(3.0 * x + 7.0, W)["estimate"] == pytest.approx(
        morans_i(x, W)["estimate"]
    )


def test_morisp_constant_x_is_undefined_not_zero():
    with pytest.raises(ValueError, match="constant"):
        morans_i(np.ones(9), _rook(3, 3))


def test_morisp_rejects_empty_weights_and_shape_mismatch():
    with pytest.raises(ValueError, match="w.. = 0"):
        morans_i(np.arange(4.0), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="must have shape"):
        morans_i(np.arange(4.0), np.zeros((3, 3)))
