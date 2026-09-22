"""Tests for fisher_z.fisher_z."""

from morie.fn import _array_core as np

from morie.fn.fisher_z import fisher_z


def test_ca11e12_basic():
    """Test basic functionality against the documented formula."""
    r = 0.5
    result = fisher_z(r)

    assert isinstance(result, dict)
    # The headline key is 'value' per the docstring.
    assert "value" in result
    assert "method" in result

    # Independent computation of Zr = 0.5 * ln((1+r)/(1-r)).
    expected_value = 0.5 * np.log((1 + r) / (1 - r))
    assert np.isclose(result["value"], expected_value)


def test_ca11e12_edge():
    """Test edge cases with scalar correlations strictly between -1 and 1."""
    # Values must be scalars in the open interval (-1, 1) so the
    # implementation's `if not -1 < r < 1` check is unambiguous.
    r_zero = 0.0
    r_pos = 0.9
    r_neg = -0.75

    res_zero = fisher_z(r_zero)
    res_pos = fisher_z(r_pos)
    res_neg = fisher_z(r_neg)

    for res in (res_zero, res_pos, res_neg):
        assert isinstance(res, dict)
        assert "value" in res

    # At r=0, Fisher's Z is exactly 0.
    assert np.isclose(res_zero["value"], 0.5 * np.log((1 + r_zero) / (1 - r_zero)))
    assert res_zero["value"] == 0.0

    # Positive correlation -> positive Z.
    assert np.isclose(res_pos["value"], 0.5 * np.log((1 + r_pos) / (1 - r_pos)))
    assert res_pos["value"] > 0

    # Negative correlation -> negative Z; antisymmetry check Z(-r) = -Z(r).
    assert np.isclose(res_neg["value"], 0.5 * np.log((1 + r_neg) / (1 - r_neg)))
    assert res_neg["value"] < 0
    assert np.isclose(res_neg["value"], -0.5 * np.log((1 + 0.75) / (1 - 0.75)))
