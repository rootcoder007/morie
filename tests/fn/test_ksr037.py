"""Tests for ksr037 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr037 import kosorok_ch2_glivenko_cantelli_uniform


def test_ksr037_basic():
    out = kosorok_ch2_glivenko_cantelli_uniform(lambda e: (1 / e) ** 2, 1.5)
    assert out["conditions_met"] is True


def test_ksr037_edge():
    # GC needs BOTH finite entropy and an integrable envelope
    bad = kosorok_ch2_glivenko_cantelli_uniform(lambda e: (1 / e) ** 2, np.inf)
    assert bad["entropy_finite"] is True and bad["conditions_met"] is False
