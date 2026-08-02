"""Tests for volrs."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volrs import vol_rogers_satchell


def test_volrs_basic():
    out = vol_rogers_satchell([100.0], [110.0], [95.0], [105.0])
    expect = np.log(110 / 105) * np.log(110 / 100) + np.log(95 / 105) * np.log(95 / 100)
    assert out["sigma2"][0] == pytest.approx(expect)


def test_volrs_edge():
    assert vol_rogers_satchell([10.0], [10.0], [10.0], [10.0])["sigma2"][0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        vol_rogers_satchell([100.0], [99.0], [95.0], [105.0])
