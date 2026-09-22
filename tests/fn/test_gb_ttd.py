"""Tests for gb_ttd.gibbons_total_runs_dist_table."""

import math

from morie.fn import _array_core as np

from morie.fn.gb_ttd import gibbons_total_runs_dist_table


def test_gb_ttd_basic():
    """Test basic functionality: distribution over the full support."""
    n1 = 8
    n2 = 7
    n = n1 + n2
    result = gibbons_total_runs_dist_table(n1, n2)

    assert isinstance(result, dict)
    # Returned keys per docstring
    assert "support" in result
    assert "pmf" in result
    assert "cdf" in result
    assert "pmf_r" in result
    assert "cdf_r" in result
    assert "sf_r" in result
    assert "mean" in result
    assert "var" in result
    assert "n1" in result
    assert "n2" in result
    assert "method" in result

    # Support shape: 2..n inclusive
    assert result["support"] == list(range(2, n + 1))
    assert len(result["pmf"]) == n - 1
    assert len(result["cdf"]) == n - 1

    # PMF sums to 1 (normalised by C(n, n1))
    assert math.isclose(sum(result["pmf"]), 1.0, rel_tol=1e-12, abs_tol=1e-12)

    # CDF is non-decreasing and ends at 1
    cdf = result["cdf"]
    for a, b in zip(cdf, cdf[1:]):
        assert b >= a - 1e-15
    assert math.isclose(cdf[-1], 1.0, rel_tol=1e-12, abs_tol=1e-12)

    # Independent recomputation of the pmf for a couple of support points
    den = math.comb(n, n1)
    pmf_indep = []
    for rr in range(2, n + 1):
        if rr % 2 == 0:
            k = rr // 2
            p = 2.0 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        else:
            k = (rr - 1) // 2
            p = (
                math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k)
                + math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
            )
        pmf_indep.append(p / den)

    for got, expected in zip(result["pmf"], pmf_indep):
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)

    # Mean and variance derived independently
    s = result["support"]
    p = result["pmf"]
    mean_indep = sum(ss * pp for ss, pp in zip(s, p))
    ex2_indep = sum(ss * ss * pp for ss, pp in zip(s, p))
    var_indep = ex2_indep - mean_indep * mean_indep
    assert math.isclose(result["mean"], mean_indep, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["var"], var_indep, rel_tol=1e-12, abs_tol=1e-12)
    assert result["n1"] == n1
    assert result["n2"] == n2


def test_gb_ttd_at_r():
    """Test querying the pmf/cdf/sf at a specific r."""
    n1 = 5
    n2 = 6
    n = n1 + n2
    r = 7
    result = gibbons_total_runs_dist_table(n1, n2, r)

    assert isinstance(result, dict)
    # pmf_r, cdf_r, sf_r should now be finite numbers
    assert isinstance(result["pmf_r"], float)
    assert isinstance(result["cdf_r"], float)
    assert isinstance(result["sf_r"], float)
    assert math.isfinite(result["pmf_r"])
    assert math.isfinite(result["cdf_r"])
    assert math.isfinite(result["sf_r"])

    # Recompute independently using the docstring formula
    den = math.comb(n, n1)
    support = list(range(2, n + 1))
    pmf = []
    for rr in support:
        if rr % 2 == 0:
            k = rr // 2
            val = 2.0 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        else:
            k = (rr - 1) // 2
            val = (
                math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k)
                + math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
            )
        pmf.append(val / den)
    idx = r - 2
    cdf_indep = []
    acc = 0.0
    for pp in pmf:
        acc += pp
        cdf_indep.append(acc)
    sf_indep = 1.0 - (cdf_indep[idx - 1] if idx > 0 else 0.0)

    assert math.isclose(result["pmf_r"], pmf[idx], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["cdf_r"], cdf_indep[idx], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["sf_r"], sf_indep, rel_tol=1e-12, abs_tol=1e-12)
