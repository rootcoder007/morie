"""Tests for scmaba.synthetic_control_method."""

from morie.fn import _array_core as np
import pytest

from morie.fn.scmaba import synthetic_control_method


def test_scmaba_basic():
    rng = np.random.default_rng(42)
    T, J, t0 = 40, 8, 25
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    Y0 = f @ rng.uniform(0, 1, size=(2, J)) + rng.normal(scale=0.1, size=(T, J))
    y1 = 0.5 * Y0[:, 0] + 0.5 * Y0[:, 1] + rng.normal(scale=0.1, size=T)
    y1[t0:] += 4.0
    result = synthetic_control_method(y1, Y0, t0)
    assert result["att"] == pytest.approx(4.0, abs=0.5)  # measured ~4.0
    assert result["rmse_pre"] < 0.5
    assert result["weights"].sum() == pytest.approx(1.0, abs=1e-6)


def test_scmaba_edge():
    with pytest.raises(ValueError):
        synthetic_control_method([1.0, 2.0, 3.0], np.ones((3, 2)), 1)  # t0 < 2
    with pytest.raises(ValueError):
        synthetic_control_method([1.0, 2.0, 3.0], np.ones((4, 2)), 2)  # T mismatch
