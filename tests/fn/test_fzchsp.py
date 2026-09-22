"""Tests for fzchsp.fauzi_chung_smirnov."""

from morie.fn import _array_core as np

from morie.fn.fzchsp import fauzi_chung_smirnov


def test_fzchsp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    cdf = lambda t: 0.5 * (1.0 + np.sign(t) * (1.0 - np.exp(-2.0 * t * t / np.pi)))
    h = 0.3
    result = fauzi_chung_smirnov(x, cdf, h=h)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "supdiff" in result
    assert "scale" in result
    assert "h" in result
    assert "n" in result
    assert "method" in result

    # Recompute expected statistic independently from the documented formula.
    from scipy.stats import norm as _norm
    xv = np.asarray(x, dtype=float).ravel()
    n = int(xv.size)
    g = np.sort(xv)
    khat = np.asarray(
        [float(np.mean(_norm.cdf((float(t) - xv) / h))) for t in g], dtype=float
    )
    fv = np.asarray([float(cdf(float(t))) for t in g], dtype=float)
    sup = float(np.max(np.abs(khat - fv)))
    scale = float(np.sqrt(2.0 * n / np.log(np.log(n))))
    expected_stat = scale * sup
    expected_sup = sup
    expected_scale = scale

    assert result["n"] == n
    assert abs(result["h"] - h) < 1e-12
    assert abs(result["scale"] - expected_scale) < 1e-12
    assert abs(result["supdiff"] - expected_sup) < 1e-12
    assert abs(result["statistic"] - expected_stat) < 1e-12


def test_fzchsp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    cdf = lambda t: 0.5 * (1.0 + np.sign(t) * (1.0 - np.exp(-2.0 * t * t / np.pi)))
    h = 0.3
    result = fauzi_chung_smirnov(x, cdf, h=h)
    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["statistic"] >= 0.0
    assert result["supdiff"] >= 0.0
    assert result["scale"] > 0.0
    assert result["method"] == "Chung-Smirnov normalised uniform error of the KDFE"
