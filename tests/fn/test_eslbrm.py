"""Tests for eslbrm.esl_boltzmann."""

from morie.fn import _array_core as np

from morie.fn.eslbrm import esl_boltzmann


def test_eslbrm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(44)
    n = 100
    d = 5
    v = (rng.random((n, d)) < 0.5).astype(float)
    h = 3
    result = esl_boltzmann(v, h=h, n_epochs=10, seed=1)

    # Returns a RichResult that supports dict-like access.
    assert hasattr(result, "__getitem__") or isinstance(result, dict)

    # The documented keys must be present.
    for key in (
        "W", "a", "b", "hidden_prob", "reconstruction",
        "reconstruction_error", "error_path", "free_energy",
    ):
        assert key in result

    # Shapes match the documented contract.
    assert result["W"].shape == (d, h)
    assert result["a"].shape == (d,)
    assert result["b"].shape == (h,)
    assert result["hidden_prob"].shape == (n, h)
    assert result["reconstruction"].shape == (n, d)
    assert result["error_path"].shape == (10,)
    assert result["free_energy"].shape == (n,)

    # Probabilities are probabilities; reconstruction is binary-ish in [0,1].
    hp = result["hidden_prob"]
    assert hp.min() >= 0.0 and hp.max() <= 1.0
    recon = result["reconstruction"]
    assert recon.min() >= 0.0 and recon.max() <= 1.0

    # error_path length matches n_epochs; reconstruction_error matches its last entry.
    assert float(result["reconstruction_error"]) == float(result["error_path"][-1])


def test_eslbrm_edge():
    """Test edge cases."""
    # Tiny but valid binary input: a single repeated pattern.
    v = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    result = esl_boltzmann(v, h=2, n_epochs=2, seed=0)
    for key in (
        "W", "a", "b", "hidden_prob", "reconstruction",
        "reconstruction_error", "error_path", "free_energy",
    ):
        assert key in result
    assert result["W"].shape == (3, 2)
    assert result["error_path"].shape == (2,)
