"""Tests for bnssel.bound_selection."""

from morie.fn import _array_core as np

from morie.fn.bnssel import bound_selection


def test_bnssel_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng = np.random.default_rng(42)
    n = 100

    # Discrete stratum label: 4 strata from a multinomial-like draw.
    X = rng.choice([0, 1, 2, 3], size=n)

    # Selection indicator must be coded 0/1, with rates that vary by stratum.
    base_p = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
    D = np.array([1.0 if rng.random() < base_p[int(x)] else 0.0
                  for x in X])

    # Outcome; the recorded value for D == 0 is ignored by the function.
    y = rng_y.normal(0.0, 1.0, n)

    result = bound_selection(y, D, X)

    # The implementation returns a RichResult (mapping-like) dict.
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate",
                "p_observed", "n_strata", "n"):
        assert key in result

    # Independent recomputation of the worst-case bound per the docstring
    # formula, using only the observed (D == 1) outcomes and stratum info.
    y0 = min(yv for yv, dv in zip(y, D) if dv == 1.0)
    y1 = max(yv for yv, dv in zip(y, D) if dv == 1.0)

    strata = sorted({int(x) for x in X})
    lo = 0.0
    hi = 0.0
    for g in strata:
        idx = [i for i in range(n) if int(X[i]) == g]
        ng = len(idx)
        obs_y = [y[i] for i in idx if D[i] == 1.0]
        n1 = len(obs_y)
        m1 = sum(obs_y) / n1
        p1 = n1 / ng
        a_lo = m1 * p1 + y0 * (1.0 - p1)
        a_hi = m1 * p1 + y1 * (1.0 - p1)
        wgt = ng / n
        lo += wgt * a_lo
        hi += wgt * a_hi

    assert abs(result["lower"] - lo) < 1e-12
    assert abs(result["upper"] - hi) < 1e-12
    assert abs(result["width"] - (hi - lo)) < 1e-12
    assert abs(result["estimate"] - 0.5 * (lo + hi)) < 1e-12

    n1_total = int(sum(D))
    assert abs(result["p_observed"] - n1_total / n) < 1e-12
    assert result["n"] == n
    assert result["n_strata"] == len(strata)
    assert result["lower"] <= result["upper"]


def test_bnssel_edge():
    """Test edge cases: identical selection rates collapse to pooled bound."""
    rng_y = np.random.default_rng(43)
    rng = np.random.default_rng(42)
    n = 100

    # Same selection rate across strata: stratified bound == pooled bound.
    X = rng.choice([0, 1, 2, 3], size=n)
    D = np.array([1.0 if rng.random() < 0.6 else 0.0 for _ in range(n)])

    y = rng_y.normal(0.0, 1.0, n)

    result = bound_selection(y, D, X)

    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate",
                "p_observed", "n_strata", "n"):
        assert key in result

    # Independent pooled-bound computation (single stratum).
    obs_y = [yv for yv, dv in zip(y, D) if dv == 1.0]
    n1 = len(obs_y)
    p1 = n1 / n
    m1 = sum(obs_y) / n1
    y0 = min(obs_y)
    y1 = max(obs_y)
    lo = m1 * p1 + y0 * (1.0 - p1)
    hi = m1 * p1 + y1 * (1.0 - p1)

    assert abs(result["lower"] - lo) < 1e-12
    assert abs(result["upper"] - hi) < 1e-12
    assert result["n"] == n
    assert result["n_strata"] == 4
