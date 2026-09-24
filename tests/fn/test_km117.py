"""Tests for km117.kamath_ch8_bleu_final."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km117 import kamath_ch8_bleu_final


def test_km117_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    BP = 0.5  # brevity penalty in [0, 1]
    # n-gram precisions: each in [0, 1]
    p_n = [float(rng.uniform(0.1, 0.9)) for _ in range(4)]
    result = kamath_ch8_bleu_final(BP, p_n)
    assert isinstance(result, dict)
    # Check all expected keys are present
    for key in ("estimate", "brevity_penalty", "geometric_mean",
                "p_n", "n", "method"):
        assert key in result
    # estimate = BP * geometric_mean (structural identity from the formula)
    assert math.isclose(result["estimate"],
                        result["brevity_penalty"] * result["geometric_mean"])
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["geometric_mean"])
    # geometric mean of values in [0, 1] must itself lie in [0, 1]
    assert 0.0 <= result["geometric_mean"] <= 1.0
    assert result["brevity_penalty"] == BP


def test_km117_edge():
    """Test edge cases."""
    # BP = 1 (no brevity penalty) with a single precision:
    # estimate must equal that single precision exactly.
    result = kamath_ch8_bleu_final(1.0, [0.5])
    assert isinstance(result, dict)
    assert math.isclose(result["estimate"], 0.5)
    assert result["brevity_penalty"] == 1.0

    # Passing N explicitly (matching len(p_n)) still yields a finite
    # BLEU score in [0, 1].
    result2 = kamath_ch8_bleu_final(0.5, [0.5, 0.25], N=2)
    assert isinstance(result2, dict)
    assert math.isfinite(result2["estimate"])
    assert 0.0 <= result2["estimate"] <= 1.0

    # The docstring states BP must lie in [0, 1]; values outside must
    # raise ValueError.
    with pytest.raises(ValueError):
        kamath_ch8_bleu_final(1.5, [0.5, 0.25])
    with pytest.raises(ValueError):
        kamath_ch8_bleu_final(-0.1, [0.5, 0.25])
