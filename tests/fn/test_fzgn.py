"""Tests for fzgn.fauzi_gn_edgeworth_correction."""

from morie.fn import _array_core as np

from morie.fn.fzgn import fauzi_gn_edgeworth_correction


def test_fzgn_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 100)
    n = 200
    h = 0.3
    sigma = 0.5
    e1, e2, e3, e4, e5, e6 = 0.1, -0.2, 0.05, 0.02, -0.03, 0.04
    delta = 0.0
    result = fauzi_gn_edgeworth_correction(
        x, n, h, sigma, e1, e2, e3, e4, e5, e6, delta=delta, book=False
    )

    # Documented keys
    for key in ("estimate", "normal", "correction", "book", "method"):
        assert key in result

    assert result["book"] is False
    assert isinstance(result["method"], str)

    # Independent recomputation of G_n(x) from the documented formula
    from morie.fn import _stats_core as stats

    xv = np.asarray(x, dtype=float) - delta / (sigma * np.sqrt(n))
    phi = stats.norm.pdf(xv)
    base = stats.norm.cdf(xv)
    he2 = xv ** 2 - 1.0
    he3 = xv ** 3 - 3.0 * xv
    he5 = xv ** 5 - 10.0 * xv ** 3 + 15.0 * xv
    term1 = he2 / (6.0 * np.sqrt(n) * sigma ** 3) * (e1 + 3.0 * e2 / h)
    inner = (
        xv / (4.0 * sigma ** 2) * (4.0 * e5 + e6)
        + he3 / (6.0 * sigma ** 4) * (3.0 * e3 + e4)
        + he5 / (8.0 * sigma ** 6) * e2 ** 2
    )
    corr = phi * (term1 + inner / (n * h * h))
    expected_estimate = base - corr
    expected_normal = base
    expected_correction = corr

    assert np.allclose(result["estimate"], expected_estimate)
    assert np.allclose(result["normal"], expected_normal)
    assert np.allclose(result["correction"], expected_correction)


def test_fzgn_edge():
    """Test edge cases: book=True uses 3 e_{2n} + e_{4n} in the He_3 bracket."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 50)
    n = 150
    h = 0.4
    sigma = 0.7
    e1, e2, e3, e4, e5, e6 = 0.11, 0.22, 0.33, 0.44, 0.55, 0.66

    result_book = fauzi_gn_edgeworth_correction(
        x, n, h, sigma, e1, e2, e3, e4, e5, e6, book=True
    )
    result_paper = fauzi_gn_edgeworth_correction(
        x, n, h, sigma, e1, e2, e3, e4, e5, e6, book=False
    )

    assert isinstance(result_book, dict)
    assert isinstance(result_paper, dict)
    assert result_book["book"] is True
    assert result_paper["book"] is False

    # The two spellings should differ whenever e2 != e3 (here they do)
    assert not np.allclose(result_book["estimate"], result_paper["estimate"])
