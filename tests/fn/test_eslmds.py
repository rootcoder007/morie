"""Tests for eslmds.esl_mds."""

import math

from morie.fn import _array_core as np

from morie.fn.eslmds import esl_mds


def _build_line_distance_matrix(n):
    """Build a valid symmetric, zero-diagonal Euclidean distance matrix
    from n points on a line using plain arithmetic (independent of the
    function under test)."""
    coords = np.arange(n, dtype=float)
    diffs = coords[:, None] - coords[None, :]
    return np.abs(diffs)


def test_eslmds_basic():
    """Test basic functionality on a Euclidean, line-embedded dataset."""
    D = _build_line_distance_matrix(10)
    k = 5
    result = esl_mds(D, k)

    # The function is documented to return a dict with these keys.
    assert isinstance(result, dict)
    expected_keys = {
        "estimate", "coordinates", "eigenvalues", "negative_eigenvalue_mass",
        "is_euclidean", "stress", "n", "k", "method",
    }
    assert expected_keys.issubset(result.keys())

    assert result["n"] == 10
    assert result["k"] == k

    # Line data is fully Euclidean, so no negative eigenvalues should be reported.
    assert result["is_euclidean"] is True
    assert result["negative_eigenvalue_mass"] == 0.0

    # For line-embedded data embedded in 1D, the reconstruction must be
    # exact (distance-preserving) up to numerical tolerance.
    Z = np.asarray(result["coordinates"]).reshape(10, k)
    rec = np.sqrt(np.maximum(
        np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2), 0.0))
    assert bool(np.allclose(rec, D, atol=1e-9))

    # Stress is normalised by sum(D**2); the formula below mirrors the
    # implementation independently and must agree.
    denom = float(np.sum(D ** 2))
    expected_stress = math.sqrt(float(np.sum((D - rec) ** 2)) / denom) if denom > 0 else 0.0
    assert math.isclose(result["stress"], expected_stress, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(result["estimate"], expected_stress, rel_tol=1e-9, abs_tol=1e-12)


def test_eslmds_edge():
    """Test edge cases: a non-Euclidean dissimilarity is flagged."""
    # Tree metric on a star: one point far from both others, but the two
    # leaves are close. This is NOT Euclidean, so negative eigenvalues must
    # be reported and is_euclidean must be False.
    bad = np.asarray(
        [[0.0, 1.0, 9.0],
         [1.0, 0.0, 1.0],
         [9.0, 1.0, 0.0]]
    )
    result = esl_mds(bad, 1)

    assert isinstance(result, dict)
    assert result["is_euclidean"] is False
    assert result["negative_eigenvalue_mass"] > 0.0
    assert result["n"] == 3
    assert result["k"] == 1
