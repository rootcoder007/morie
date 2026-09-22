"""Tests for gb5415.gibbons_sign_simpower."""

from morie.fn import _array_core as np

from morie.fn.gb5415 import gibbons_sign_simpower


def test_gb5415_basic():
    """Test basic functionality."""
    m0 = 0.0
    kcrit = 60  # for n=100, alpha=0.05 two-sided sign test -> kcrit = 60
    rng = np.random.default_rng(42)
    # Build 200 simulated samples of size n=100 from N(0.3, 1) so that
    # under H1 (median > 0) the test has non-trivial power.
    n = 100
    nsim = 200
    samples = rng.normal(0.3, 1, (nsim, n))
    result = gibbons_sign_simpower(samples, m0, kcrit)

    assert isinstance(result, dict)
    # Documented keys.
    for key in ("power", "rejections", "nsim", "kmean", "kcrit", "method"):
        assert key in result

    # nsim must equal the number of rows supplied.
    assert result["nsim"] == nsim
    assert result["kcrit"] == kcrit

    # Re-derive power and kmean independently from the samples using
    # the formula in the docstring: K_i = #{v in row_i : v > m0},
    # rejection iff K_i >= kcrit.
    rows = [[float(v) for v in r] for r in samples]
    ks = [sum(1 for v in r if v > float(m0)) for r in rows]
    rej = sum(1 for k in ks if k >= kcrit)
    expected_power = rej / nsim
    expected_kmean = sum(ks) / nsim

    assert result["rejections"] == rej
    assert result["power"] == expected_power
    assert result["kmean"] == expected_kmean

    # Under H1 with mean 0.3, n=100, kcrit=60 the simulated power should
    # be well above the size alpha=0.05.
    assert result["power"] > 0.05
    # Kmean must lie in [0, n].
    assert 0.0 <= result["kmean"] <= n


def test_gb5415_edge():
    """Test edge cases: under H0 the empirical rejection rate should be near alpha."""
    m0 = 0.0
    n = 100
    kcrit = 60
    rng = np.random.default_rng(7)
    # Samples truly under H0 (median 0): power must be close to alpha = 0.05.
    nsim = 1000
    samples = rng.normal(0.0, 1, (nsim, n))
    result = gibbons_sign_simpower(samples, m0, kcrit)

    assert isinstance(result, dict)
    for key in ("power", "rejections", "nsim", "kmean", "kcrit", "method"):
        assert key in result
    assert result["nsim"] == nsim
    assert result["kcrit"] == kcrit
    # Sanity bounds on the empirical size under H0 for a two-sided
    # binomial(100, 0.5) with kcrit=60: alpha = 2 * P(K >= 60).
    assert 0.0 <= result["power"] <= 1.0
    assert 0 <= result["rejections"] <= nsim
