"""Tests for evdec.evt_declustering_runs."""

from morie.fn import _array_core as np

from morie.fn.evdec import evt_declustering_runs


def test_evdec_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_u = np.random.default_rng(44)
    x = rng_x.normal(0, 1, 100)
    # Threshold must be a scalar float per the docstring of
    # evt_declustering_runs. The original test passed a length-100 array
    # of random values, which is rejected by float().
    u = float(rng_u.standard_normal())
    r = 10
    result = evt_declustering_runs(x, u, r)

    # Independent computation of expected return values from the formula
    # (Smith 1989): cluster = consecutive exceedances within gap r.
    x_arr = np.asarray(x)
    exceed = x_arr > u
    n = len(x_arr)
    cur = 0
    gap = r + 1
    expected_cid = [0] * n
    for i in range(n):
        if exceed[i]:
            if gap > r:
                cur += 1
            gap = 0
            expected_cid[i] = cur
        else:
            gap += 1
    expected_n_clusters = cur
    expected_n_exceed = sum(1 for v in expected_cid if v > 0)
    expected_theta = (
        expected_n_clusters / float(expected_n_exceed)
        if expected_n_exceed else float("nan")
    )

    assert isinstance(result, dict)
    # Documented keys must all be present.
    for key in (
        "cluster_max", "cluster_id", "n_clusters", "theta",
        "n_exceed", "estimate", "n",
    ):
        assert key in result

    # Core numeric quantities computed independently above.
    assert result["n"] == n
    assert result["n_clusters"] == expected_n_clusters
    assert result["n_exceed"] == expected_n_exceed
    assert result["theta"] == expected_theta
    assert result["estimate"] == expected_theta
    assert result["cluster_id"] == expected_cid
    assert len(result["cluster_max"]) == expected_n_clusters


def test_evdec_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_u = np.random.default_rng(44)
    x = rng_x.normal(0, 1, 100)
    u = float(rng_u.standard_normal())
    r = 10
    result = evt_declustering_runs(x, u, r)
    assert isinstance(result, dict)
    assert "n_clusters" in result
    assert "estimate" in result
