"""Tests for ca5e2.ca_chapter_5_equation_2."""

from morie.fn import _array_core as np

from morie.fn.ca5e2 import ca_chapter_5_equation_2


def test_ca5e2_basic():
    """Test basic functionality: scalar xb_m returns a dict with 'value' key."""
    rng = np.random.default_rng(42)
    xb_m = float(rng.normal(0, 1))
    result = ca_chapter_5_equation_2(xb_m)

    assert isinstance(result, dict)
    assert "value" in result
    # Per the docstring the function returns the (scalar) logit value,
    # implemented as a plain float pass-through of xb_m.
    assert float(result["value"]) == float(xb_m)
    assert result["value"] == xb_m


def test_ca5e2_zero():
    """Edge case: xb_m = 0 yields logit = 0 (equal log-probabilities)."""
    xb_m = 0.0
    result = ca_chapter_5_equation_2(xb_m)

    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
    # Independent recomputation of the documented formula:
    # logit(y=m|x) = xb_m, so the expected value is just xb_m.
    expected = xb_m
    assert float(result["value"]) == expected


def test_ca5e2_positive_and_negative():
    """Edge case: positive and negative scalar xb_m values are passed through."""
    for xb_m in (1.5, -2.25, 7.0):
        result = ca_chapter_5_equation_2(xb_m)

        assert isinstance(result, dict)
        assert "value" in result
        # The formula is the identity on xb_m; compute expected independently.
        expected = float(xb_m)
        assert float(result["value"]) == expected
