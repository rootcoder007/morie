"""Tests for bndmoq.bound_moment_qed."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.bndmoq import bound_moment_qed


def _make_D(rng, n, p):
    """Build a length-n 0/1 observation indicator with Bernoulli(p)."""
    return (rng.random(n) < p).astype(int)


def _type1_quantile(sorted_vals, level):
    """Reference implementation of the type-1 (inverse empirical CDF) quantile.

    Given a sorted ascending array ``sorted_vals`` and a probability ``level``
    in [0, 1], return the smallest index i such that the empirical CDF
    exceeds ``level``, then return ``sorted_vals[i]``.  This matches the
    convention used by ``B.q1`` inside the function under test.
    """
    m = len(sorted_vals)
    # k = ceil(m * level), then take the (k-1)-th element (1-indexed -> 0-indexed).
    import math
    k = max(1, math.ceil(m * level))
    return sorted_vals[k - 1]


def _band_for(ys, ds, a):
    """Independent recomputation of the pooled bound for the given slice.

    Mirrors the documented expressions::

        lower = q(1 - (1 - alpha) / p1)   if p1 > 1 - alpha,  else y0
        upper = q(alpha / p1)             if p1 >= alpha,      else y1
    """
    y0, y1 = min(ys), max(ys)
    obs = sorted([float(ys[i]) for i in range(len(ys)) if ds[i] == 1.0])
    m = len(ys)
    p1 = len(obs) / float(m)
    if not obs:
        return y0, y1, p1
    if p1 > 1.0 - a:
        lo = _type1_quantile(obs, 1.0 - (1.0 - a) / p1)
    else:
        lo = y0
    if p1 >= a:
        hi = _type1_quantile(obs, a / p1)
    else:
        hi = y1
    return lo, hi, p1


def test_bndmoq_basic():
    """Test basic functionality with valid, documented inputs."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_X = np.random.default_rng(7)

    n = 100
    y = rng_y.normal(0.0, 1.0, n)
    # D must be coded 0/1.
    D = (rng_D.random(n) < 0.7).astype(int)
    # X is described as a "discrete stratum label, one per unit"; give
    # each unit a small integer stratum id.
    X = rng_X.integers(0, 4, n)
    # quantile is a float in (0, 1).
    quantile = 0.3

    result = bound_moment_qed(y, D, X, quantile)

    # The function returns a RichResult (dict-like); keys are documented.
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate",
                "max_width", "n_strata", "p_observed", "n"):
        assert key in result, "missing documented key: " + key

    # Numeric expectations computed independently from the documented formula.
    lo_exp, hi_exp, p1_exp = _band_for(list(y), [float(v) for v in D], quantile)
    assert result["lower"] == lo_exp
    assert result["upper"] == hi_exp
    assert result["width"] == hi_exp - lo_exp
    assert result["estimate"] == 0.5 * (lo_exp + hi_exp)
    assert result["p_observed"] == p1_exp
    assert result["n"] == n

    # n_strata is the number of distinct stratum labels in X.
    assert result["n_strata"] == len(set(int(v) for v in X))


def test_bndmoq_edge():
    """Test an edge case: all units observed, single stratum."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)

    n = 100
    y = rng_y.normal(0.0, 1.0, n)
    # All observed -> p1 = 1, so the bound collapses onto the support of y
    # when alpha <= 1.  Choose a small alpha and a single stratum.
    D = np.ones(n, dtype=int)
    X = np.zeros(n, dtype=int)
    quantile = 0.1

    result = bound_moment_qed(y, D, X, quantile)

    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate",
                "max_width", "n_strata", "p_observed", "n"):
        assert key in result, "missing documented key: " + key

    # p1 is 1.0 because every unit is observed.
    assert result["p_observed"] == 1.0
    assert result["n"] == n
    assert result["n_strata"] == 1
    # With one stratum the widest within-stratum interval equals the pooled one.
    assert result["max_width"] == result["width"]

    # Independent recomputation: when p1 = 1 both quantile levels collapse to
    # alpha and 1 - alpha respectively.
    lo_exp, hi_exp, _ = _band_for(list(y), [float(v) for v in D], quantile)
    assert result["lower"] == lo_exp
    assert result["upper"] == hi_exp
