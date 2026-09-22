"""Tests for gh_c12_3.ghosal_strong_apx_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_3 import ghosal_strong_apx_dp


def test_gh_c12_3_basic():
    """Test basic functionality."""
    result = ghosal_strong_apx_dp(n=3000, seed=42)
    assert "estimate" in result
    est = result["estimate"]
    arr = np.asarray(est, dtype=float)
    assert arr.size == 1
    assert np.all(np.isfinite(arr))

    # Independent recomputation of the KS-style statistic from the
    # documented strong-approximation formula:
    #   sup = max_i max(|(i+1)/n - v_i|, |i/n - v_i|)
    #   ks  = sqrt(n) * sup
    rng = np.random.default_rng(42)
    data = sorted(float(rng.uniform(0, 1)) for _ in range(3000))
    n = 3000
    sup = 0.0
    for i, v in enumerate(data):
        d1 = abs((i + 1) / n - v)
        d2 = abs(i / n - v)
        if d1 > sup:
            sup = d1
        if d2 > sup:
            sup = d2
    import math
    expected_ks = math.sqrt(n) * sup
    assert float(arr) == expected_ks


def test_gh_c12_3_edge():
    """Test edge cases."""
    result = ghosal_strong_apx_dp(n=1, seed=0)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert est.size == 1
    assert np.all(np.isfinite(est))
