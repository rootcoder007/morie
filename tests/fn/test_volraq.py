"""Tests for volraq."""

import numpy as np
import pytest

from morie.fn.volraq import vol_realised_quadratic_var


def test_volraq_basic():
    out = vol_realised_quadratic_var([0.0, 1.0, 3.0])
    assert out["qv"] == pytest.approx(1 + 4)
    assert out["rq"] == pytest.approx(2 / 3 * (1 + 16))


def test_volraq_edge():
    with pytest.raises(ValueError):
        vol_realised_quadratic_var([1.0, 2.0])
