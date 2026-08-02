"""Tests for weakid.weak_identification_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.weakid import weak_identification_mediation


def test_weakid_basic():
    out = weak_identification_mediation(0.5, 0.5, 0.05, 0.05)
    assert out["weakly_identified"] is False
    assert out["ab"] == pytest.approx(0.25)
    assert out["sobel_se"] == pytest.approx(np.sqrt(2 * 0.5**2 * 0.05**2))


def test_weakid_edge():
    weak = weak_identification_mediation(0.5, 0.05, 0.05, 0.05)
    assert weak["weak_b"] is True and weak["weakly_identified"] is True
    with pytest.raises(ValueError):
        weak_identification_mediation(1.0, 1.0, 0.0, 0.1)
