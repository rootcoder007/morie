"""Tests for aitvar.aitchison_variation."""

from morie.fn import _array_core as np

from morie.fn.aitvar import aitchison_variation


def test_aitvar_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 10.0, (100, 5))
    result = aitchison_variation(X)

    # Compute the expected variation matrix and totvar from the formula
    n, D = X.shape
    lr = np.log(X[:, :, None] / X[:, None, :])
    # Sample variance with denominator n-1
    expected_tau = ((lr - lr.mean(axis=0)) ** 2).sum(axis=0) / (n - 1)
    expected_totvar = float(expected_tau[np.triu_indices(D, k=1)].sum() / D)

    assert isinstance(result, dict)
    assert "variation" in result
    assert "totvar" in result
    assert "n" in result
    assert "D" in result

    variation = np.asarray(result["variation"])
    assert variation.shape == (D, D)

    # Zero diagonal
    assert np.allclose(np.diag(variation), 0.0)

    # Symmetric
    assert np.allclose(variation, variation.T)

    # Matches the formula
    assert np.allclose(variation, expected_tau, atol=1e-10)

    # totvar = (1/D) sum_{i<j} tau_ij
    assert np.isclose(result["totvar"], expected_totvar, atol=1e-10)
    assert np.isclose(
        result["totvar"],
        float(np.sum(np.triu(variation, k=1)) / D),
        atol=1e-10,
    )

    assert result["n"] == n
    assert result["D"] == D


def test_aitvar_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.5, 5.0, (50, 4))
    result = aitchison_variation(X)

    n, D = X.shape
    assert isinstance(result, dict)
    assert "variation" in result
    assert "totvar" in result
    assert "n" in result
    assert "D" in result

    variation = np.asarray(result["variation"])
    assert variation.shape == (D, D)
    assert np.allclose(np.diag(variation), 0.0)
    assert np.allclose(variation, variation.T)

    # Independent computation of totvar from the formula
    lr = np.log(X[:, :, None] / X[:, None, :])
    expected_tau = ((lr - lr.mean(axis=0)) ** 2).sum(axis=0) / (n - 1)
    expected_totvar = float(expected_tau[np.triu_indices(D, k=1)].sum() / D)
    assert np.isclose(result["totvar"], expected_totvar, atol=1e-10)

    assert result["n"] == n
    assert result["D"] == D
