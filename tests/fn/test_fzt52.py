"""Tests for fzt52.fauzi_thm5_2_bdfree_kdfe_bv."""

from morie.fn import _array_core as np

from morie.fn.fzt52 import fauzi_thm5_2_bdfree_kdfe_bv


def test_fzt52_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    bandwidth = 0.3
    # Pick a single evaluation point inside the support.
    x0 = 0.0
    fx = 0.5  # F_X(0) for a standard normal
    density = float(np.exp(-x0 ** 2 / 2.0) / np.sqrt(2.0 * np.pi))
    c1 = 0.25  # arbitrary documented coefficient c1(x)
    dg = 1.0   # g'(g^{-1}(x)) for identity g
    result = fauzi_thm5_2_bdfree_kdfe_bv(
        n=n,
        h=bandwidth,
        fx=fx,
        density=density,
        c1=c1,
        dg=dg,
    )
    assert isinstance(result, dict)

    # Verify returned keys match the documented RichResult schema.
    for key in ("bias", "variance", "se", "edfvar", "vargain", "h", "n", "method"):
        assert key in result

    # Independent recomputation of the documented formulas (Gaussian r1 ~= 1/(2*sqrt(pi))).
    mu2 = 1.0
    r1 = 1.0 / (2.0 * np.sqrt(np.pi))
    bias = (bandwidth ** 2 / 2.0) * c1 * mu2
    edfvar = fx * (1.0 - fx) / n
    vargain = 2.0 * bandwidth / n * dg * density * r1
    variance = edfvar - vargain

    assert result["bias"] == (bandwidth ** 2 / 2.0) * c1 * mu2
    assert result["edfvar"] == fx * (1.0 - fx) / n
    assert result["vargain"] == 2.0 * bandwidth / n * dg * density * r1
    assert result["variance"] == result["edfvar"] - result["vargain"]
    assert result["n"] == int(n)
    assert result["h"] == bandwidth


def test_fzt52_edge():
    """Test edge cases: Gaussian r1 default and positive variance."""
    n = 200
    bandwidth = 0.5
    x0 = 0.5
    fx = 0.5
    density = float(np.exp(-x0 ** 2 / 2.0) / np.sqrt(2.0 * np.pi))
    c1 = 0.1
    dg = 1.0
    result = fauzi_thm5_2_bdfree_kdfe_bv(
        n=n,
        h=bandwidth,
        fx=fx,
        density=density,
        c1=c1,
        dg=dg,
    )
    assert isinstance(result, dict)
    # Gaussian default r1 must be used (positive): vargain > 0.
    assert result["vargain"] > 0.0
    # Variance smaller than naive empirical-df variance when dg >= 1.
    assert result["variance"] < result["edfvar"]
