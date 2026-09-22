"""Tests for gh_c4_15.ghosal_dp_mutual_sing."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_15 import ghosal_dp_mutual_sing


def test_gh_c4_15_basic():
    """Test basic functionality: equal continuous parts and equal atom sets -> not singular."""
    # Continuous parts (mass on a grid) - identical here
    cont_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cont_2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Atom support-point lists - identical here
    atoms_1 = np.array([0.0, 1.0])
    atoms_2 = np.array([0.0, 1.0])

    result = ghosal_dp_mutual_sing(cont_1, cont_2, atoms_1, atoms_2)

    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation of the formula:
    # continuous parts differ iff lengths differ or any |u-v| > 1e-12
    cont_differ = (len(cont_1) != len(cont_2)) or any(
        abs(float(u) - float(v)) > 1e-12 for u, v in zip(cont_1, cont_2)
    )
    # atomic supports differ iff the sets of points are not equal
    atoms_differ = set(float(a) for a in atoms_1) != set(float(a) for a in atoms_2)
    singular = cont_differ or atoms_differ
    expected_estimate = 1.0 if singular else 0.0

    assert float(result["estimate"]) == expected_estimate
    assert bool(result["mutually_singular"]) == singular
    assert bool(result["continuous_parts_differ"]) == cont_differ
    assert bool(result["atomic_supports_differ"]) == atoms_differ


def test_gh_c4_15_edge():
    """Test edge cases: single-element inputs, continuous parts equal, atom supports equal."""
    cont_1 = np.array([42.0])
    cont_2 = np.array([42.0])
    atoms_1 = np.array([7.0])
    atoms_2 = np.array([7.0])

    result = ghosal_dp_mutual_sing(cont_1, cont_2, atoms_1, atoms_2)

    # Independent computation of the formula:
    cont_differ = (len(cont_1) != len(cont_2)) or any(
        abs(float(u) - float(v)) > 1e-12 for u, v in zip(cont_1, cont_2)
    )
    atoms_differ = set(float(a) for a in atoms_1) != set(float(a) for a in atoms_2)
    singular = cont_differ or atoms_differ
    expected_estimate = 1.0 if singular else 0.0

    assert float(result["estimate"]) == expected_estimate
    assert bool(result["mutually_singular"]) == singular
    assert bool(result["continuous_parts_differ"]) == cont_differ
    assert bool(result["atomic_supports_differ"]) == atoms_differ
