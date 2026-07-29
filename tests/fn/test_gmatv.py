"""Tests for gmatv.grm_vanraden.

The generated versions of these tests called the function with a 1-D
vector, which it correctly refuses -- markers are an (n x m) genotype
matrix. Expectations here are derived from VanRaden (2008) Eq. 3
directly, not from running the function:
Z = M - 2P with P_j = p_j the column allele frequency, and
G = Z Z' / (2 sum_j p_j (1 - p_j)).
"""

import numpy as np
import pytest

from morie.fn.gmatv import grm_vanraden


def test_gmatv_basic():
    """G is symmetric and matches the hand-written VanRaden formula."""
    M = np.array([[0.0, 1.0, 2.0],
                  [1.0, 1.0, 0.0],
                  [2.0, 0.0, 1.0],
                  [0.0, 2.0, 2.0]])
    result = grm_vanraden(M)
    G = np.asarray(result["estimate"], dtype=float)
    assert G.shape == (4, 4)
    assert np.allclose(G, G.T)
    p = M.mean(axis=0) / 2.0
    Z = M - 2.0 * p
    expected = Z @ Z.T / (2.0 * np.sum(p * (1.0 - p)))
    assert np.allclose(G, expected)
    assert np.allclose(np.asarray(result["p"], dtype=float), p)
    assert result["n"] == 4
    assert result["m"] == 3


def test_gmatv_centering_makes_offdiagonals_negative_on_average():
    """Centring by 2p forces the mean off-diagonal relationship to 0-.

    Sum of all Z columns is zero by construction, so sum(G) = 0 and the
    off-diagonal mean is exactly -(diagonal mean)/(n-1).
    """
    M = np.array([[0.0, 1.0, 2.0],
                  [1.0, 1.0, 0.0],
                  [2.0, 0.0, 1.0],
                  [0.0, 2.0, 2.0]])
    result = grm_vanraden(M)
    n = result["n"]
    assert np.isclose(result["off_mean"], -result["diag_mean"] / (n - 1))


def test_gmatv_edge():
    """A 1-D argument is refused: markers must be a 2-D (n x m) array."""
    with pytest.raises(ValueError, match="2D"):
        grm_vanraden(np.array([42.0]))
