"""Tests for cthrgr.causal_three_layer_grf."""

from morie.fn import _array_core as np

from morie.fn.cthrgr import causal_three_layer_grf


def test_cthrgr_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_m = np.random.default_rng(41)
    rng_x = np.random.default_rng(40)
    n = 100
    y = rng_y.normal(0, 1, n)
    # Binary treatment: draw uniform then threshold at 0.5
    D = (rng_d.uniform(0, 1, n) > 0.5).astype(int)
    M = rng_m.normal(0, 1, n)
    X = rng_x.normal(0, 1, (n, 5))
    result = causal_three_layer_grf(y, D, M, X)
    assert isinstance(result, dict)
    # Documented keys in the returned RichResult payload.
    assert "estimate" in result
    assert "direct" in result
    assert "indirect" in result
    assert "total" in result
    assert "propensity" in result
    assert "nde" in result
    assert "nie" in result
    assert "proportion_mediated" in result
    assert "overlap_min" in result
    assert "overlap_max" in result
    # Shapes of per-query arrays must match the number of query points
    # (here equal to n since newX is None).
    assert len(result["direct"]) == n
    assert len(result["indirect"]) == n
    assert len(result["total"]) == n
    assert len(result["propensity"]) == n
    # Scalar averages equal sums divided by n.
    nde_avg = sum(result["direct"]) / n
    nie_avg = sum(result["indirect"]) / n
    tot_avg = sum(result["total"]) / n
    assert abs(result["nde"] - nde_avg) < 1e-12
    assert abs(result["nie"] - nie_avg) < 1e-12
    assert abs(result["estimate"] - tot_avg) < 1e-12
    # The total effect per point equals direct + indirect.
    for d_i, i_i, t_i in zip(result["direct"], result["indirect"], result["total"]):
        assert abs((d_i + i_i) - t_i) < 1e-9
    # Overlap bounds bracket the propensity values.
    assert min(result["propensity"]) >= result["overlap_min"]
    assert max(result["propensity"]) <= result["overlap_max"]


def test_cthrgr_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_m = np.random.default_rng(41)
    rng_x = np.random.default_rng(40)
    n = 100
    y = rng_y.normal(0, 1, n)
    D = (rng_d.uniform(0, 1, n) > 0.5).astype(int)
    M = rng_m.normal(0, 1, n)
    X = rng_x.normal(0, 1, (n, 5))
    result = causal_three_layer_grf(y, D, M, X, route="mean", n_draw=1)
    assert isinstance(result, dict)
    # With n_draw==1 the residual sweep collapses to the median residual,
    # which is 0.0 in the source, so n_draw in the payload is 1.
    assert result["n_draw"] == 1
    assert "estimate" in result
