"""Tests for gh_c14_21.ghosal_ord_dep_sbp."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_21 import ghosal_ord_dep_sbp


def test_gh_c14_21_basic():
    """Test basic functionality.

    The function operates on a scalar x with a tuple/list of atom
    locations and a tuple/list of stick weights V (all documented as
    1-D sequences, V the same length as atom_locs). We compute the
    expected weight by hand following the docstring formula: sort atoms
    by distance to x, then walk the sorted order, multiplying the
    remaining probability (1 - V[rank] for ranks already taken) into
    each stick.
    """
    x = 0.3
    atom_locs = (0.1, 0.35, 0.6, 0.9)
    V = (0.5, 0.5, 0.5, 0.5)

    # Distances from x to each atom:
    #   |0.1-0.3|=0.2, |0.35-0.3|=0.05, |0.6-0.3|=0.3, |0.9-0.3|=0.6
    # Sorted closest-first: 0.35 (rank 0), 0.1 (rank 1), 0.6 (rank 2), 0.9 (rank 3)
    # Indices in that order: 1, 0, 2, 3
    # V in that order:        0.5, 0.5, 0.5, 0.5
    # W at each rank r (in the sorted order):
    #   W[rank 0] = 1.0 * 0.5                       = 0.5
    #   W[rank 1] = 0.5 * 0.5                       = 0.25
    #   W[rank 2] = 0.25 * 0.5                      = 0.125
    #   W[rank 3] = 0.125 * 0.5                     = 0.0625
    # The estimate is W at the nearest atom (index 1, rank 0) = 0.5.
    # The full W array, placed back into the original atom order:
    #   W[0]=0.25, W[1]=0.5, W[2]=0.125, W[3]=0.0625
    expected_estimate = 0.5
    expected_weights = (0.25, 0.5, 0.125, 0.0625)

    result = ghosal_ord_dep_sbp(x=x, atom_locs=atom_locs, V=V)

    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert est == expected_estimate

    # The function also returns the full weight vector.
    assert "weights" in result
    got_weights = tuple(float(v) for v in np.asarray(result["weights"], dtype=float))
    assert got_weights == expected_weights

    # Nearest atom dominates by construction with positive V.
    assert result["nearest_dominates"] is True

    # And the method string is the documented citation.
    assert result["method"].startswith("ordering-dependent sticks")

    # estimate must be the maximum weight (and therefore finite).
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c14_21_edge():
    """Test that the function evaluates on a single-atom configuration.

    The docstring defines atom_locs and V as sequences of equal length
    (not scalars), so a single-atom case has length-1 sequences.
    """
    x = 42.0
    atom_locs = (0.0,)
    V = (0.5,)

    # With one atom and one stick of 0.5, W = [1.0 * 0.5] = [0.5],
    # nearest atom is the only one, so estimate = 0.5.
    expected_estimate = 1.0 * 0.5

    result = ghosal_ord_dep_sbp(x=x, atom_locs=atom_locs, V=V)

    est = float(np.asarray(result["estimate"], dtype=float))
    assert est == expected_estimate
    assert float(np.asarray(result["weights"], dtype=float)[0]) == expected_estimate
    assert result["nearest_dominates"] is True
