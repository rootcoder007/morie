"""Tests for aittvr.aitchison_total_variance."""

from morie.fn import _array_core as np

from morie.fn.aittvr import aitchison_total_variance


def test_aittvr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.lognormal(mean=0.0, sigma=1.0, size=(100, 5))
    result = aitchison_total_variance(X)

    assert isinstance(result, dict)
    for key in ("totvar", "totvar_trace", "clr_var", "n", "D"):
        assert key in result

    n, D = X.shape
    assert result["n"] == n
    assert result["D"] == D

    # Independent computation via the documented formula:
    # totvar = trace(Gamma) = (1/D) * sum_{i<j} var{log(x_i/x_j)}
    # with sample variance (ddof=1).
    L = np.log(X)
    pairwise_sum = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            diff = L[:, i] - L[:, j]
            pairwise_sum += np.var(diff, ddof=1)
    expected_totvar = pairwise_sum / D

    assert np.isclose(result["totvar"], expected_totvar)
    assert np.isclose(result["totvar_trace"], expected_totvar)

    # The two equivalent forms must agree.
    assert np.isclose(result["totvar"], result["totvar_trace"])

    # clr_var has one entry per component.
    assert len(result["clr_var"]) == D


def test_aittvr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.lognormal(mean=0.0, sigma=1.0, size=(100, 5))
    result = aitchison_total_variance(X)
    assert isinstance(result, dict)
    assert "totvar" in result
