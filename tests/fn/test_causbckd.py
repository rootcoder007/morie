"""Tests for causbckd.causal_backdoor_estimate."""

from morie.fn import _array_core as np

from morie.fn.causbckd import causal_backdoor_estimate


def _make_data():
    """Build a deterministic, well-formed dataset.

    y: continuous outcome, length n.
    X: binary treatment (0/1), length n.
    Z: discrete stratum labels (hashable), length n.
    """
    rng = np.random.default_rng(43)
    n = 100
    y = rng.normal(0.0, 1.0, n)
    Z = ["s1", "s2", "s3"] * (n // 3) + ["s1"] * (n - 3 * (n // 3))
    X_raw = rng.uniform(0.0, 1.0, n)
    X = [1 if X_raw[i] < 0.5 + 0.2 * (Z[i] == "s2") - 0.1 * (Z[i] == "s3") else 0
         for i in range(n)]
    return [float(v) for v in y], X, Z


def test_causbckd_basic():
    """Test basic functionality and the documented result keys."""
    y, X, Z = _make_data()
    result = causal_backdoor_estimate(y, X, Z)

    # Documented return: a dict-like RichResult payload with these keys.
    assert "estimate" in result
    assert "se" in result
    assert "strata" in result
    assert "n" in result

    assert result["n"] == len(y)

    # Compute the ATE independently from the docstring's formula and
    # compare against the function's estimate.
    n = len(y)
    Z_set = sorted(set(Z), key=lambda v: str(v))
    ate_expected = 0.0
    se_sq_expected = 0.0
    for k in Z_set:
        i1 = [i for i in range(n) if Z[i] == k and X[i] == 1]
        i0 = [i for i in range(n) if Z[i] == k and X[i] == 0]
        y1 = [y[i] for i in i1]
        y0 = [y[i] for i in i0]
        m1 = sum(y1) / len(y1)
        m0 = sum(y0) / len(y0)
        d = m1 - m0
        w = (len(i1) + len(i0)) / n
        ate_expected += w * d
        # Within-arm sample variance (ddof = 1).
        v1 = sum((v - m1) ** 2 for v in y1) / (len(y1) - 1) if len(y1) > 1 else 0.0
        v0 = sum((v - m0) ** 2 for v in y0) / (len(y0) - 1) if len(y0) > 1 else 0.0
        se_sq_expected += w * w * (v1 / len(y1) + v0 / len(y0))
    se_expected = se_sq_expected ** 0.5

    assert abs(result["estimate"] - ate_expected) < 1e-12
    assert abs(result["se"] - se_expected) < 1e-12

    # Each stratum entry must carry share, effect, n1, n0.
    for k, s in result["strata"].items():
        assert "share" in s
        assert "effect" in s
        assert "n1" in s
        assert "n0" in s
        assert s["n1"] > 0 and s["n0"] > 0  # positivity


def test_causbckd_edge():
    """Test edge cases: same well-formed shapes, both arms populated."""
    y, X, Z = _make_data()
    result = causal_backdoor_estimate(y, X, Z)
    assert isinstance(result, dict)
    assert result["n"] == len(y)
