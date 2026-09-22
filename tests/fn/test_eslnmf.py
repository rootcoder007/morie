"""Tests for eslnmf.esl_nmf."""

from morie.fn import _array_core as np

from morie.fn.eslnmf import esl_nmf


def test_eslnmf_basic():
    """Test basic functionality on non-negative data."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5)) ** 2  # square to ensure non-negativity
    k = 5
    result = esl_nmf(X, k)
    assert isinstance(result, dict)
    for key in ("estimate", "W", "H", "frobenius_error", "relative_error",
                "iterations", "converged", "n", "p", "k", "method"):
        assert key in result
    # W and H must be non-negative (multiplicative updates preserve this)
    assert min(result["W"]) >= 0.0
    assert min(result["H"]) >= 0.0
    # Shapes must match the documented (n, k) and (k, p)
    assert len(result["W"]) == result["n"] * result["k"] == 100 * k
    assert len(result["H"]) == result["k"] * result["p"] == k * 5
    # Reconstruction: relative Frobenius error equals ||X - W H||_F / ||X||_F
    W = np.asarray(result["W"]).reshape(result["n"], result["k"])
    H = np.asarray(result["H"]).reshape(result["k"], result["p"])
    normX = float(np.linalg.norm(X)) or 1.0
    expected_rel_err = float(np.linalg.norm(X - W @ H)) / normX
    assert abs(result["relative_error"] - expected_rel_err) < 1e-12
    assert abs(result["estimate"] - expected_rel_err) < 1e-12
    assert abs(result["frobenius_error"] - float(np.linalg.norm(X - W @ H))) < 1e-12


def test_eslnmf_edge():
    """Test edge cases with exactly rank-1 non-negative data."""
    # Rank-1 outer product should be recovered essentially exactly.
    X = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
    k = 1
    result = esl_nmf(X, k)
    assert isinstance(result, dict)
    assert result["k"] == k
    assert result["n"] == 3 and result["p"] == 2
    assert min(result["W"]) >= 0.0 and min(result["H"]) >= 0.0
    # Independent reconstruction check
    W = np.asarray(result["W"]).reshape(3, 1)
    H = np.asarray(result["H"]).reshape(1, 2)
    expected_rel_err = float(np.linalg.norm(X - W @ H)) / (float(np.linalg.norm(X)) or 1.0)
    assert abs(result["relative_error"] - expected_rel_err) < 1e-12
    # The reconstruction matches the data even though W, H are only
    # determined up to a positive scaling.
    assert bool(np.allclose(W @ H, X, atol=1e-5))
