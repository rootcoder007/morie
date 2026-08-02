"""Tests for stquo."""

from morie.fn import _array_core as np
import pytest

from morie.fn.stquo import status_quo_spatial


def test_stquo_basic():
    ideal = [0.0, 1.0, 2.0, 3.0, 10.0]
    out = status_quo_spatial(ideal, status_quo=0.0, proposal=2.5)
    assert out["passes"] is True
    assert out["votes_for"] == 3


def test_stquo_edge():
    assert status_quo_spatial([0.0, 1.0], 2.0, 5.0)["passes"] is False
    with pytest.raises(ValueError):
        status_quo_spatial([0.0, 1.0], [0.0, 0.0], 1.0)  # dimension mismatch
