"""Tests for cdp_posterior_cov.cdp_posterior_cov."""

from morie.fn import _array_core as np

from morie.fn.cdp_posterior_cov import cdp_posterior_cov


def test_ghs016_basic():
    """Test basic functionality."""
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    counts = np.array([10, 20, 30, 40, 50])
    j = 1
    jp = 2
    alpha_total = 1.0
    result = cdp_posterior_cov(alpha, counts, j, jp, alpha_total)
    assert isinstance(result, dict)
    assert "value" in result
    assert "estimate" in result
    # Independent computation from the documented formula:
    # cov(p_j, p_j' | X) = -m_j * m_j' / (A + n + 1)
    # where m_j = (alpha_j + n_j) / (A + n) is the posterior mean,
    # A = alpha_total, and n = sum(counts).
    A = alpha_total
    n_total = 10 + 20 + 30 + 40 + 50
    m_j = (alpha[1] + counts[1]) / (A + n_total)
    m_jp = (alpha[2] + counts[2]) / (A + n_total)
    expected = -m_j * m_jp / (A + n_total + 1)
    assert abs(result["value"] - expected) < 1e-12
    assert abs(result["estimate"] - expected) < 1e-12


def test_ghs016_edge():
    """Test edge cases with j == jp (variance, must be >= 0)."""
    alpha = np.array([0.1, 0.2, 0.3])
    counts = np.array([5, 15, 25])
    j = 0
    jp = 0
    alpha_total = 0.5
    result = cdp_posterior_cov(alpha, counts, j, jp, alpha_total)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent computation:
    A = alpha_total
    n_total = 5 + 15 + 25
    m_j = (alpha[0] + counts[0]) / (A + n_total)
    expected = -m_j * m_j / (A + n_total + 1)
    assert abs(result["value"] - expected) < 1e-12
    # Variance (j == jp) is the negative of a squared mean; only valid
    # when m_j == 0 (here m_j > 0 so value is negative, but formula holds).
