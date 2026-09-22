"""Tests for gh_ap_f2.ghosal_glivenko."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_f2 import ghosal_glivenko


def test_gh_ap_f2_basic():
    """Test basic functionality: empirical KS distance shrinks with n."""
    result = ghosal_glivenko()
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "sup_by_n" in result
    assert "vanishing" in result
    assert "method" in result

    # The estimate must be the last entry of sup_by_n, and all entries
    # must be finite non-negative floats (KS distances).
    sup_by_n = result["sup_by_n"]
    estimate = result["estimate"]
    assert len(sup_by_n) == 3  # default ns = (100, 1000, 10000)
    assert estimate == sup_by_n[-1]

    arr = np.asarray(sup_by_n, dtype=float)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)

    # Independent re-derivation of the formula on the SAME inputs that
    # the function must have used internally: for n in {100, 1000, 10000},
    # draw a uniform sample with the same RNG (default seed=42) and
    # compute max_i max(|F_n(v_i) - v_i|, |F_n(v_{i-1}) - v_i|) where
    # F_n is the empirical CDF of the sample.
    rng = np.random.default_rng(42)
    expected_sups = []
    for n in (100, 1000, 10000):
        data = sorted(float(rng.uniform(0, 1)) for _ in range(n))
        sup = 0.0
        for i, v in enumerate(data):
            sup = max(sup,
                      abs((i + 1) / n - v),
                      abs(i / n - v))
        expected_sups.append(sup)

    got = np.asarray(sup_by_n, dtype=float)
    want = np.asarray(expected_sups, dtype=float)
    assert np.all(np.abs(got - want) < 1e-12)

    # Documented behaviour: "vanishing" is True iff the last sup is
    # smaller than the first sup AND smaller than 0.02.
    assert result["vanishing"] is bool(
        sup_by_n[-1] < sup_by_n[0] and sup_by_n[-1] < 0.02
    )


def test_gh_ap_f2_edge():
    """Test that a custom (small) ns tuple and custom seed are honoured."""
    result = ghosal_glivenko(ns=(50,), seed=0)

    assert "estimate" in result
    assert "sup_by_n" in result
    assert len(result["sup_by_n"]) == 1
    assert result["estimate"] == result["sup_by_n"][0]

    # Independent re-derivation for the single n=50 sample.
    rng = np.random.default_rng(0)
    data = sorted(float(rng.uniform(0, 1)) for _ in range(50))
    sup = 0.0
    for i, v in enumerate(data):
        sup = max(sup,
                  abs((i + 1) / 50 - v),
                  abs(i / 50 - v))
    assert abs(float(result["sup_by_n"][0]) - sup) < 1e-12

    # The function does not return an "n" key; the documented keys are
    # estimate, sup_by_n, vanishing, method. Assert that explicitly.
    assert "n" not in result
    assert isinstance(result["method"], str)
