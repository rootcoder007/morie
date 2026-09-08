"""Tests for b102.burkov_lm_ch1_squared_error."""

from morie.fn.b102 import burkov_lm_ch1_squared_error


def test_b102_basic():
    assert burkov_lm_ch1_squared_error(3.0, 1.0)["estimate"] == 4.0


def test_b102_edge():
    import pytest
    with pytest.raises(ValueError, match="same shape"):
        burkov_lm_ch1_squared_error([1.0, 2.0], [1.0])
