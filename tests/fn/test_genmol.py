"""Tests for genmol.generative_chemistry."""

from morie.fn import _array_core as np

from morie.fn.genmol import generative_chemistry


def test_genmol_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    model = {
        "mu": rng.normal(0.0, 1.0, 8).tolist(),
        "logvar": rng.normal(-1.0, 0.5, 8).tolist(),
    }
    n_samples = 5
    conditions = {
        "property": lambda z: sum(zi * zi for zi in z),
        "training_set": ["CCO", "CCN"],
    }
    result = generative_chemistry(model, n_samples, conditions)
    assert isinstance(result, dict)
    assert "smiles" in result
    assert "validity" in result
    assert "uniqueness" in result
    assert "novelty" in result
    assert "kl" in result
    assert "n_samples" in result
    assert result["n_samples"] == 5


def test_genmol_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    model = {
        "mu": rng.normal(0.0, 1.0, 4).tolist(),
        "logvar": rng.normal(-1.0, 0.5, 4).tolist(),
    }
    n_samples = 3
    conditions = {}
    result = generative_chemistry(model, n_samples, conditions)
    assert isinstance(result, dict)
    assert result["n_samples"] == 3
    assert "smiles" in result
    assert "validity" in result
