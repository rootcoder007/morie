"""Tests for gb_fwt.gibbons_fligner_wolfe_test."""

import math

from morie.fn import _array_core as np

from morie.fn.gb_fwt import gibbons_fligner_wolfe_test


def test_gb_fwt_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)

    # Two groups: control (samples[0]) and one treatment group.
    ctrl = rng.normal(0.0, 1.0, 20)
    treat_a = rng.normal(0.5, 1.0, 20)
    samples = [ctrl, treat_a]

    result = gibbons_fligner_wolfe_test(samples)

    # The function returns a dict-like object (RichResult).
    assert hasattr(result, "__getitem__") or isinstance(result, dict)

    # ---- Independent recomputation of every reported quantity ----
    n1 = 20
    mt = 20
    rr = (n1 // 2) + 1          # default median index
    t = sorted([float(v) for v in ctrl])[rr - 1]
    w_indep = sum(1 for v in [float(v) for v in treat_a] if v < t)
    den = math.comb(n1 + mt, mt)
    pmf_indep = [
        math.comb(n1 + mt - rr - j, mt - j) * math.comb(rr + j - 1, j) / den
        for j in range(mt + 1)
    ]
    pvalue_indep = min(1.0, sum(pmf_indep[: w_indep + 1]))
    mean_indep = sum(j * p for j, p in enumerate(pmf_indep))

    # ---- Existing loose oracle-style assertions ----
    assert "statistic" in result
    assert "p_value" in result

    # ---- Independent equality checks ----
    assert int(result["statistic"]) == w_indep
    assert result["t"] == float(t)
    assert result["r"] == rr
    assert result["mtreat"] == mt
    assert result["k"] == 2

    assert math.isclose(result["p_value"], pvalue_indep, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result["mean"], mean_indep, rel_tol=1e-12, abs_tol=1e-15)

    # pmf has length mt + 1 and sums to ~1.
    pmf = result["pmf"]
    assert len(pmf) == mt + 1
    assert math.isclose(sum(pmf), 1.0, abs_tol=1e-12)
    for p in pmf_indep:
        assert any(math.isclose(p, q, abs_tol=1e-15) for q in pmf)


def test_gb_fwt_edge():
    """Test edge cases with custom r."""
    rng = np.random.default_rng(42)

    ctrl = rng.normal(0.0, 1.0, 15)
    treat_a = rng.normal(0.0, 1.0, 10)
    treat_b = rng.normal(0.0, 1.0, 10)
    samples = [ctrl, treat_a, treat_b]

    r = 3
    result = gibbons_fligner_wolfe_test(samples, r=r)

    assert isinstance(result, dict) or hasattr(result, "__getitem__")

    # Independent recomputation with k=3 treatment observations pooled.
    n1 = 15
    mt = 20
    ctrl_sorted = sorted([float(v) for v in ctrl])
    t = ctrl_sorted[r - 1]
    all_treat = [float(v) for v in treat_a] + [float(v) for v in treat_b]
    w_indep = sum(1 for v in all_treat if v < t)
    den = math.comb(n1 + mt, mt)
    pmf_indep = [
        math.comb(n1 + mt - r - j, mt - j) * math.comb(r + j - 1, j) / den
        for j in range(mt + 1)
    ]
    pvalue_indep = min(1.0, sum(pmf_indep[: w_indep + 1]))

    assert int(result["statistic"]) == w_indep
    assert result["r"] == r
    assert result["mtreat"] == mt
    assert result["k"] == 3
    assert math.isclose(result["p_value"], pvalue_indep, rel_tol=1e-12, abs_tol=1e-15)
    assert result["t"] == float(t)
