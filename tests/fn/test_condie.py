"""Tests for condie.conditional_indirect_effect."""

from morie.fn import _array_core as np

from morie.fn.condie import conditional_indirect_effect


def test_condie_basic():
    """Test basic functionality with scalar coefficients and an array of w values."""
    rng_a1 = np.random.default_rng(42)
    rng_a3 = np.random.default_rng(43)
    rng_b = np.random.default_rng(44)
    rng_w = np.random.default_rng(45)

    a1 = float(rng_a1.normal(0, 1))
    a3 = float(rng_a3.normal(0, 1))
    b = float(rng_b.normal(0, 1))
    w = rng_w.exponential(1, 25)

    sa1 = 0.1
    sa3 = 0.2
    sb = 0.3
    sa1a3 = 0.05

    result = conditional_indirect_effect(a1, a3, b, w, sa1=sa1, sa3=sa3, sa1a3=sa1a3, sb=sb)

    # The function returns a RichResult which should behave like a dict for keys.
    assert isinstance(result, dict)

    # Required documented keys must be present.
    for key in (
        "estimate",
        "simple_slope",
        "se",
        "se_slope",
        "z",
        "p_value",
        "w",
        "n",
        "method",
    ):
        assert key in result, f"Missing key: {key}"

    # The function returns arrays when w has more than one element.
    assert result["n"] == int(np.asarray(w).ravel().size)
    assert result["method"].startswith("Conditional indirect effect")

    # Independently compute the expected quantities from the formula:
    #   slope(w) = a1 + a3 * w
    #   effect(w) = b * slope(w)
    wv = np.asarray(w, dtype=float).ravel()
    simple_slope = a1 + a3 * wv
    estimate = b * simple_slope

    vslope = sa1 ** 2 + 2.0 * sa1a3 * wv + sa3 ** 2 * wv ** 2
    se_slope = np.sqrt(vslope)
    se = np.sqrt(simple_slope ** 2 * sb ** 2 + b ** 2 * vslope)

    # Mean of effects is what the function reports as `estimate`.
    assert np.allclose(np.asarray(result["estimate"]), float(np.mean(estimate)))
    # Array of effects is stored under the `effect` key.
    assert np.allclose(np.asarray(result["effect"]), estimate)
    assert np.allclose(np.asarray(result["simple_slope"]), simple_slope)
    assert np.allclose(np.asarray(result["se_slope"]), se_slope)
    assert np.allclose(np.asarray(result["se"]), se)
    assert np.allclose(np.asarray(result["w"]), wv)


def test_condie_edge():
    """Test scalar w (single moderator value) returns scalars."""
    a1 = 0.4
    a3 = 0.25
    b = 0.6
    w = 1.5
    sa1 = 0.1
    sa3 = 0.2
    sb = 0.3
    sa1a3 = 0.05

    result = conditional_indirect_effect(a1, a3, b, w, sa1=sa1, sa3=sa3, sa1a3=sa1a3, sb=sb)

    assert isinstance(result, dict)
    for key in (
        "estimate",
        "simple_slope",
        "se",
        "se_slope",
        "z",
        "p_value",
        "w",
        "n",
        "method",
    ):
        assert key in result, f"Missing key: {key}"

    # Scalar branch: n is exactly 1.
    assert result["n"] == 1
    assert result["w"] == float(w)

    # Independent computation of expected scalars.
    simple_slope = a1 + a3 * float(w)
    estimate = b * simple_slope
    vslope = sa1 ** 2 + 2.0 * sa1a3 * float(w) + sa3 ** 2 * float(w) ** 2
    se_slope = float(np.sqrt(vslope))
    se = float(np.sqrt(simple_slope ** 2 * sb ** 2 + b ** 2 * vslope))

    assert result["simple_slope"] == simple_slope
    assert result["estimate"] == estimate
    assert result["se_slope"] == se_slope
    assert result["se"] == se
