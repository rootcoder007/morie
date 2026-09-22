"""Tests for btiid.boot_iid_resample."""

from morie.fn import _array_core as np

from morie.fn.btiid import boot_iid_resample


def _statistic(data):
    """A simple statistic: the sample mean."""
    return float(np.mean(data))


def test_btiid_basic():
    """Test basic functionality of Efron's nonparametric bootstrap."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    B = 1000
    result = boot_iid_resample(x, _statistic, B=B, seed=0)

    # The function returns a RichResult (mapping-like) carrying the
    # bootstrap replicates and summary diagnostics.
    assert hasattr(result, "__getitem__")
    assert "estimate" in result
    assert "replicates" in result

    # The on-original estimate is simply the statistic evaluated on x.
    expected_estimate = float(np.mean(x))
    assert float(result["estimate"]) == expected_estimate

    # Bias of the bootstrap distribution is mean(replicates) - estimate,
    # both of which we recompute independently from the returned replicates.
    reps = result["replicates"]
    expected_bias = float(np.mean(reps)) - expected_estimate
    assert float(result["bias"]) == expected_bias

    # The standard error is the sample standard deviation of the replicates.
    expected_se = float(np.std(reps, ddof=1))
    assert float(result["se"]) == expected_se

    # The percentile CI is the 2.5% / 97.5% quantiles of the replicates.
    lo, hi = np.percentile(reps, [2.5, 97.5])
    expected_ci = (float(lo), float(hi))
    assert tuple(map(float, result["ci_percentile"])) == expected_ci

    # B and n reflect the actual replicate count and sample size used.
    assert int(result["B"]) == B
    assert int(result["n"]) == x.shape[0]

    # The replicates array has the requested length.
    assert reps.shape[0] == B


def test_btiid_edge():
    """Test edge cases: deterministic input with a custom statistic."""
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 2.0, 50)
    B = 500
    result = boot_iid_resample(x, _statistic, B=B, seed=123)

    assert "estimate" in result
    assert "consistency_caveat" in result
    assert result["method"] == "Efron (1979) nonparametric IID bootstrap"
    assert int(result["B"]) == B
    assert int(result["n"]) == x.shape[0]
