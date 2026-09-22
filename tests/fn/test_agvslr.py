"""Tests for agvslr.alphazero_value_lr."""

from morie.fn import _array_core as np

from morie.fn.agvslr import alphazero_value_lr


def test_agvslr_basic():
    """Test basic functionality."""
    # Scalar inputs as documented: t, T, lr_0 are floats.
    t = 5.0
    T = 10.0
    lr_0 = 0.2
    result = alphazero_value_lr(t, T, lr_0)

    # Independent computation of the documented cosine formula:
    # lr = floor + (lr_0 - floor) * 0.5 * (1 + cos(pi * t/T))
    import math
    frac = t / T
    expected_lr = 0.0 + (lr_0 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * frac))
    expected_frac = frac

    # Payload keys documented in the docstring / RichResult payload.
    assert "estimate" in result
    assert "lr" in result
    assert "frac" in result
    assert "kind" in result
    assert result["kind"] == "cosine"
    assert np.isclose(result["estimate"], expected_lr)
    assert np.isclose(result["lr"], expected_lr)
    assert np.isclose(result["frac"], expected_frac)


def test_agvslr_edge():
    """Test edge cases."""
    # t == 0 should yield lr_0 (cosine starts at 1).
    t = 0.0
    T = 10.0
    lr_0 = 0.2
    result = alphazero_value_lr(t, T, lr_0)
    assert np.isclose(result["estimate"], lr_0)
    assert np.isclose(result["lr"], lr_0)
    assert np.isclose(result["frac"], 0.0)

    # t == T should yield floor (cosine ends at -1).
    floor = 0.01
    result_end = alphazero_value_lr(T, T, lr_0, floor=floor)
    assert np.isclose(result_end["estimate"], floor)
    assert np.isclose(result_end["lr"], floor)
    assert np.isclose(result_end["frac"], 1.0)

    # AlphaZero "step" schedule kind.
    result_step = alphazero_value_lr(0.5 * T, T, lr_0, kind="step")
    assert result_step["kind"] == "step"
    # At frac = 0.5 -> idx = 2 -> _STEPS[2]/_STEPS[0]
    assert 0.0 <= result_step["estimate"] <= lr_0
