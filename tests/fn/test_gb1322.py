"""Tests for gb1322.gibbons_are_formula."""

from morie.fn import _array_core as np

from morie.fn.gb1322 import gibbons_are_formula


def test_gb1322_basic():
    """Test basic functionality against the documented ARE formula."""
    # docstring parameters are four floats:
    #   deriv, var, deriv_star, var_star
    deriv = 2.5
    var = 3.0
    deriv_star = 1.5
    var_star = 4.0

    result = gibbons_are_formula(deriv, var, deriv_star, var_star)

    # The function returns a RichResult mapping whose documented keys are
    # ``are``, ``check`` (efficacy ratio), ``efficacy``, ``efficacy_star``,
    # ``method``.
    assert isinstance(result, dict)
    for key in ("are", "check", "efficacy", "efficacy_star", "method"):
        assert key in result

    # Independent recomputation of eq. (13.2.1):
    # ARE = (d / ds)^2 * (vs / v)
    expected_are = (deriv / deriv_star) ** 2 * (var_star / var)
    assert result["are"] == expected_are

    # Efficacy e(T) = [dE/dtheta]^2 / sigma^2, computed independently for
    # both tests; ``check`` is their ratio, which must equal ARE.
    eff = deriv * deriv / var
    eff_star = deriv_star * deriv_star / var_star
    assert result["efficacy"] == eff
    assert result["efficacy_star"] == eff_star
    assert result["check"] == eff / eff_star
    assert result["check"] == expected_are


def test_gb1322_edge():
    """Test edge cases."""
    # Variances must be strictly positive and the reference derivative must
    # be non-zero per the docstring; valid scalar inputs still hit the
    # code path and return the documented mapping.
    deriv = 1.0
    var = 1.0
    deriv_star = 2.0
    var_star = 1.0

    result = gibbons_are_formula(deriv, var, deriv_star, var_star)
    assert isinstance(result, dict)
    assert "are" in result
    assert "check" in result
    assert "efficacy" in result
    assert "efficacy_star" in result

    expected_are = (deriv / deriv_star) ** 2 * (var_star / var)
    assert result["are"] == expected_are
