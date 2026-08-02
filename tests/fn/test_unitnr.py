"""Tests for unitnr.unit_nonresponse."""

from morie.fn import _array_core as np
import pytest

from morie.fn.unitnr import unit_nonresponse


def test_unitnr_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(size=2000)
    phi = 1 / (1 + np.exp(-(0.5 + 1.5 * x)))
    r = (rng.random(2000) < phi).astype(float)
    y = 2.0 + 1.5 * x + rng.normal(scale=0.5, size=2000)
    result = unit_nonresponse(r, None, x, y=y)
    # respondent mean is biased upward (measured ~2.7); weighting recovers 2
    assert abs(y[r == 1].mean() - 2.0) > 0.3
    assert result["estimate"] == pytest.approx(2.0, abs=0.3)
    assert np.all(result["weights"][r == 0] == 0.0)


def test_unitnr_edge():
    with pytest.raises(ValueError):
        unit_nonresponse([1, 1, 1], None, [1.0, 2.0, 3.0])  # all respond
    with pytest.raises(ValueError):
        unit_nonresponse([0.5, 1], None, [1.0, 2.0])  # non-binary indicator
