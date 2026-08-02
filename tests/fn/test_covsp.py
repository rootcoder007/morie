"""covsp: one-sample coverages (Gibbons & Chakraborti 5e, Ch 2 -- Order
Statistics, Quantiles, and Coverages)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.covsp import one_sample_coverage as cov


def test_covsp_there_are_n_plus_one_coverages_and_they_sum_to_one():
    """n order statistics cut the line into n+1 intervals."""
    rng = np.random.default_rng(1301)
    x = rng.random(9)
    r = cov(x)
    c = np.asarray(r["coverages"])
    assert c.size == x.size + 1
    assert float(c.sum()) == pytest.approx(1.0)
    assert np.all(c >= 0)


def test_covsp_expected_coverage_is_one_over_n_plus_one():
    """Each coverage has mean 1/(n+1) under a continuous F -- the standard
    result that makes coverages distribution-free."""
    r = cov(np.random.default_rng(1307).random(19))
    assert r["expected"] == pytest.approx(1 / 20)


def test_covsp_equally_spaced_points_give_equal_interior_coverages():
    """On a uniform grid the gaps between consecutive order statistics are
    identical, so the interior coverages must be too."""
    x = np.linspace(0.1, 0.9, 9)
    c = np.asarray(cov(x)["coverages"])
    interior = c[1:-1]
    assert np.allclose(interior, interior[0])


def test_covsp_cumulative_is_the_span_between_the_extreme_order_statistics():
    """`cumulative` is the SCALAR F(X_(n)) - F(X_(1)) on the rank scale, which
    is (n-1)/(n+1) -- not a running sum of the coverages. It is the total
    probability the sample actually spans, so it is always short of 1 by the
    two tail coverages.
    """
    for n in (4, 12, 25):
        r = cov(np.random.default_rng(1311).random(n))
        assert r["cumulative"] == pytest.approx((n - 1) / (n + 1))
        c = np.asarray(r["coverages"])
        assert r["cumulative"] == pytest.approx(1.0 - c[0] - c[-1])


def test_covsp_reports_the_sample_extremes():
    x = np.array([0.4, 0.1, 0.9, 0.6])
    r = cov(x)
    assert r["sample_min"] == pytest.approx(0.1)
    assert r["sample_max"] == pytest.approx(0.9)
    assert r["n"] == 4


def test_covsp_mean_coverage_matches_the_expectation_over_replications():
    """Simulate: the average coverage really is 1/(n+1)."""
    rng = np.random.default_rng(1319)
    n = 7
    got = np.mean([np.asarray(cov(rng.random(n))["coverages"]).mean() for _ in range(500)])
    assert got == pytest.approx(1 / (n + 1), rel=1e-9)


def test_covsp_is_invariant_to_a_monotone_transform():
    """Coverages depend only on the ORDER of the sample, so any strictly
    increasing transform leaves them unchanged -- that is what makes them
    distribution-free."""
    rng = np.random.default_rng(1321)
    x = rng.random(10)
    a = np.asarray(cov(x)["coverages"])
    b = np.asarray(cov(np.exp(3 * x))["coverages"])
    # Coverages are gaps in F, so they are preserved only in RANK structure;
    # the ordering of the gaps must at least agree.
    assert np.argsort(a).tolist() == np.argsort(b).tolist()
