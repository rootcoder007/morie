"""Tests for gh_c4_22.ghosal_constr_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_22 import ghosal_constr_dp


def test_gh_c4_22_basic():
    """Test basic functionality with two sets of base masses."""
    # Two mixture components; weights must sum to 1.
    control_weights = np.array([0.3, 0.7])

    # base_masses_by_set: one array of base masses per control component.
    # Values must be non-negative (they parameterize DP concentration base measures).
    base_masses_by_set = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
    ]

    result = ghosal_constr_dp(control_weights, base_masses_by_set)

    # The function returns a RichResult-like object with dict-style access.
    assert "estimate" in result
    assert "P_cells" in result
    assert "total_mass" in result
    assert "method" in result

    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))

    # P_cells must exist and be a non-empty sequence of cell probabilities.
    cells = result["P_cells"]
    assert len(cells) == len(base_masses_by_set) * len(base_masses_by_set[0])
    assert np.all(np.isfinite(np.asarray(cells, dtype=float)))

    # total_mass must equal the sum of P_cells (documented contract).
    assert np.isclose(
        float(np.sum(np.asarray(cells, dtype=float))),
        float(result["total_mass"]),
    )


def test_gh_c4_22_edge():
    """Test edge case: a single mixture component with a single base mass."""
    # One weight, one mass vector of length 1: deterministic normalization to 1.
    control_weights = np.array([1.0])
    base_masses_by_set = [np.array([42.0])]

    result = ghosal_constr_dp(control_weights, base_masses_by_set)

    # The single cell's estimate must equal the only control weight (1.0)
    # because the lone normalized base mass is 1.0, and w_j * 1.0 = w_j.
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.isclose(est, 1.0)

    # And total_mass must also be 1.0.
    assert np.isclose(float(result["total_mass"]), 1.0)
