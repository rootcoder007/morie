"""Tests for gb_ksc.gibbons_ks_cvm_comparison."""

from morie.fn import _array_core as np

from morie.fn.gb_ksc import gibbons_ks_cvm_comparison


def _normal_cdf(x):
    """Standard normal CDF using math.erf (no scipy dependency)."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_gb_ksc_basic():
    """Compare both statistics against values computed independently from
    the documented formulas.

    The function sorts the sample, computes deviations
        D_i = max((i+1)/n - z_i,  z_i - i/n)
    and reports:
        d     = max_i D_i
        w2    = 1/(12n) + sum_i (z_i - (2(i+1)-1)/(2n))^2
        argmax = argmax_i D_i
        share  = terms[argmax] / w2
    where z_i = F_0(x_(i)) for x sorted ascending.
    """
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 1, 100)
    xs = sorted(float(v) for v in raw)

    # Obtain z_i by mapping the actual callable F_0 over the sorted sample.
    # We import F_0 here (after sorting) so the test exercises the same
    # callable the production code does.
    F0 = _normal_cdf
    z = [float(F0(v)) for v in xs]

    n = len(xs)

    # --- independent recomputation of the documented formulas ---
    devs = [max((i + 1) / n - z[i], z[i] - i / n) for i in range(n)]
    expected_d = max(devs)
    expected_arg = devs.index(expected_d)

    terms = [
        (z[j] - (2.0 * (j + 1) - 1.0) / (2.0 * n)) ** 2 for j in range(n)
    ]
    expected_w2 = 1.0 / (12.0 * n) + sum(terms)
    expected_share = terms[expected_arg] / expected_w2

    result = gibbons_ks_cvm_comparison(xs, F0)

    # The function returns a RichResult. It must expose at minimum these
    # documented keys with the documented meanings.
    assert hasattr(result, "keys") or isinstance(result, dict)
    keys = set(result.keys()) if hasattr(result, "keys") else set(result.__dict__)
    assert {"d", "w2", "argmax", "share", "n", "method"} <= keys

    assert result["d"] == expected_d
    assert result["w2"] == expected_w2
    assert result["argmax"] == expected_arg
    assert result["share"] == expected_share
    assert result["n"] == n

    # Sanity: both statistics are non-negative; share is in [0, 1].
    assert result["d"] >= 0.0
    assert result["w2"] > 0.0
    assert 0.0 <= result["share"] <= 1.0


def test_gb_ksc_edge():
    """Edge case: a tiny sample against a known CDF.

    With a hand-picked 4-point sample and the standard-normal CDF,
    the documented formulas yield a unique, deterministic result.
    """
    F0 = _normal_cdf

    x = [-1.0, 0.0, 0.5, 2.0]
    xs = sorted(float(v) for v in x)
    z = [float(F0(v)) for v in xs]
    n = len(xs)

    devs = [max((i + 1) / n - z[i], z[i] - i / n) for i in range(n)]
    expected_d = max(devs)
    expected_arg = devs.index(expected_d)

    terms = [
        (z[j] - (2.0 * (j + 1) - 1.0) / (2.0 * n)) ** 2 for j in range(n)
    ]
    expected_w2 = 1.0 / (12.0 * n) + sum(terms)
    expected_share = terms[expected_arg] / expected_w2

    result = gibbons_ks_cvm_comparison(xs, F0)

    keys = set(result.keys()) if hasattr(result, "keys") else set(result.__dict__)
    assert {"d", "w2", "argmax", "share", "n", "method"} <= keys

    assert result["d"] == expected_d
    assert result["w2"] == expected_w2
    assert result["argmax"] == expected_arg
    assert result["share"] == expected_share
    assert result["n"] == n

    # Documented size constraint: n >= 2.  The callable must not raise.
    assert result["n"] == 4
