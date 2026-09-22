"""Tests for gh_c3_4.ghosal_stick_break_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_4 import ghosal_stick_break_def


def test_gh_c3_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_stick_break_def(x)
    assert "estimate" in result
    assert "weights" in result
    assert "atoms" in result
    assert "total_mass" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # weights and atoms must have the requested number of terms
    weights = np.asarray(result["weights"], dtype=float)
    atoms = np.asarray(result["atoms"], dtype=float)
    assert weights.shape == (100,)
    assert atoms.shape == (100,)

    # total_mass is the sum of the stick-breaking weights
    assert np.isclose(float(result["total_mass"]), float(np.sum(weights)))

    # estimate is the dot product of weights and atoms
    expected_mean = float(np.sum(weights * atoms))
    assert np.isclose(float(result["estimate"]), expected_mean)


def test_gh_c3_4_edge():
    """Test edge cases."""
    result = ghosal_stick_break_def(np.array([42.0]))
    # Function does not return a key named 'n'; instead, the number of
    # terms is documented via the n_terms parameter and reflected in the
    # shapes of weights and atoms.
    assert "weights" in result
    assert "atoms" in result
    weights = np.asarray(result["weights"], dtype=float)
    atoms = np.asarray(result["atoms"], dtype=float)
    # Default n_terms is 100; the input array shape is not used to size
    # the atom/weight arrays (it is not a documented argument shape).
    assert weights.shape == (100,)
    assert atoms.shape == (100,)
