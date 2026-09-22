"""Tests for gb5713.gibbons_wsrt_simpower."""

from morie.fn import _array_core as np

from morie.fn.gb5713 import gibbons_wsrt_simpower


def test_gb5713_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # samples: one row per simulated sample; 200 samples of size 30
    # drawn under H1 with median shifted to 0.3.
    samples = rng.normal(loc=0.3, scale=1.0, size=(200, 30)).tolist()
    m0 = 0.0
    # Two-sided alpha = 0.05 critical value for the Wilcoxon signed-rank
    # test at n = 30 (Table H, Gibbons & Chakraborti 2011).
    tcrit = 369.0

    result = gibbons_wsrt_simpower(samples, m0, tcrit)

    assert isinstance(result, dict)

    # --- keys documented by the function ---
    for key in ("power", "rejections", "nsim", "tmean", "tcrit", "method"):
        assert key in result, f"missing key {key!r} in result"

    # --- documented scalar types / ranges ---
    assert result["nsim"] == len(samples)
    assert isinstance(result["rejections"], int)
    assert 0 <= result["rejections"] <= result["nsim"]
    assert 0.0 <= result["power"] <= 1.0
    assert result["tcrit"] == tcrit
    assert isinstance(result["method"], str)

    # --- independent recomputation of the headline numbers ---
    # Power = rejections / nsim; rejections are exactly the number of
    # rows with T+ >= tcrit, counted under the documented algorithm.
    expected_rejections = 0
    t_values = []
    for row in samples:
        ds = [v - m0 for v in row if v != m0]
        n = len(ds)
        a = [abs(v) for v in ds]
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = mid
            i = j + 1
        t_plus = sum(ranks[i] for i in range(n) if ds[i] > 0.0)
        t_values.append(t_plus)
        if t_plus >= tcrit:
            expected_rejections += 1

    expected_power = expected_rejections / len(samples)
    expected_tmean = sum(t_values) / len(samples)

    assert result["rejections"] == expected_rejections
    assert result["power"] == expected_power
    assert result["tmean"] == expected_tmean


def test_gb5713_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # All samples identical to m0 -> no non-zero differences -> T+ = 0
    # for every row, so power at any positive tcrit is 0.
    samples = [[0.0] * 25 for _ in range(50)]
    m0 = 0.0
    tcrit = 5.0

    result = gibbons_wsrt_simpower(samples, m0, tcrit)

    assert isinstance(result, dict)
    for key in ("power", "rejections", "nsim", "tmean", "tcrit", "method"):
        assert key in result, f"missing key {key!r} in result"

    assert result["nsim"] == 50
    assert result["rejections"] == 0
    assert result["power"] == 0.0
    assert result["tmean"] == 0.0
    assert result["tcrit"] == tcrit
