"""Tests for ksr035 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr035 import kosorok_ch2_donsker_bracketing_integral


def test_ksr035_basic():
    out = kosorok_ch2_donsker_bracketing_integral(lambda e: (1 / e) ** 3)
    assert out["finite"] is True  # sqrt(log N) integrable for polynomial N


def test_ksr035_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_donsker_bracketing_integral(lambda e: 2.0, delta=0.0)
