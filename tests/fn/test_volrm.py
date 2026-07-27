"""Tests for volrm."""

import numpy as np
import pytest

from morie.fn.volrm import vol_riskmetrics


def test_volrm_basic():
    r = np.array([1.0, 2.0] + [0.5] * 20)
    out = vol_riskmetrics(r, lam=0.9)
    assert out["sigma2"][1] == pytest.approx(0.9 * out["sigma2"][0] + 0.1 * 1.0)
    assert out["lam"] == 0.9


def test_volrm_edge():
    assert vol_riskmetrics(np.ones(25))["lam"] == 0.94
    with pytest.raises(ValueError):
        vol_riskmetrics(np.ones(25), lam=0.0)
