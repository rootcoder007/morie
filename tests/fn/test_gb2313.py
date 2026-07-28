"""Tests for gb2313 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb2313 import gibbons_edf_joint_moment


def test_gb2313_basic():
    out = gibbons_edf_joint_moment(0.3, 0.7, 20)
    assert out["cov_edf"] == pytest.approx(0.3 * 0.3 / 20)


def test_gb2313_edge():
    with pytest.raises(ValueError):
        gibbons_edf_joint_moment(0.7, 0.3, 20)
