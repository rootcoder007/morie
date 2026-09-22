"""Tests for gb32asy.gibbons_runs_asymp_normal."""

from morie.fn import _array_core as np

from morie.fn.gb32asy import gibbons_runs_asymp_normal


def test_gb32asy_basic():
    """Test basic functionality."""
    n1 = 50
    n2 = 50
    r = 60
    result = gibbons_runs_asymp_normal(r, n1, n2)
    assert isinstance(result, dict)
    assert "z" in result
    assert "z_exact" in result
    assert "p_value" in result
    assert "mean" in result
    assert "var" in result
    assert "mean_exact" in result
    assert "var_exact" in result
    assert "lam" in result
    assert "n" in result
    assert "method" in result

    # Independent computation from the documented formula
    n = n1 + n2
    lam = n1 / float(n)
    mean = 2.0 * n * lam * (1.0 - lam)
    sd = 2.0 * (n ** 0.5) * lam * (1.0 - lam)
    me = 2.0 * n1 * n2 / float(n) + 1.0
    ve = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (float(n) ** 2 * (n - 1.0))
    expected_z = (r - mean) / sd
    expected_z_exact = (r - me) / (ve ** 0.5)

    assert result["mean"] == mean
    assert result["mean_exact"] == me
    assert result["var"] == sd * sd
    assert result["var_exact"] == ve
    assert result["lam"] == lam
    assert result["n"] == n
    assert result["z"] == expected_z
    assert result["z_exact"] == expected_z_exact
    # p_value must lie in (0, 1)
    assert 0.0 < result["p_value"] < 1.0


def test_gb32asy_edge():
    """Test edge cases: small counts and the continuity correction."""
    # Small valid case
    r = 4
    n1 = 5
    n2 = 5
    result = gibbons_runs_asymp_normal(r, n1, n2)
    assert isinstance(result, dict)
    assert "z" in result
    assert "p_value" in result
    assert result["n"] == n1 + n2
    assert result["lam"] == n1 / (n1 + n2)

    # Independently compute the uncorrected z
    n = n1 + n2
    lam = n1 / float(n)
    mean = 2.0 * n * lam * (1.0 - lam)
    sd = 2.0 * (n ** 0.5) * lam * (1.0 - lam)
    assert result["z"] == (r - mean) / sd

    # Continuity correction moves the numerator by 0.5 toward the mean
    r2 = 12
    n1b = 5
    n2b = 5
    nb = n1b + n2b
    lamb = n1b / float(nb)
    meanb = 2.0 * nb * lamb * (1.0 - lamb)
    sdb = 2.0 * (nb ** 0.5) * lamb * (1.0 - lamb)

    base = gibbons_runs_asymp_normal(r2, n1b, n2b, correct=False)
    corr = gibbons_runs_asymp_normal(r2, n1b, n2b, correct=True)

    d = r2 - meanb
    if d > 0:
        d_corr = d - 0.5
    elif d < 0:
        d_corr = d + 0.5
    else:
        d_corr = d
    expected_corr_z = d_corr / sdb

    assert corr["z"] == expected_corr_z
    assert base["z"] == d / sdb
