"""Tests for gb_ar6 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_ar6 import gibbons_are_unif


def test_gb_ar6_basic():
    assert gibbons_are_unif()["wilcoxon_vs_t"] == pytest.approx(1.0)


def test_gb_ar6_edge():
    # sign test attains the Hodges-Lehmann lower bound 1/3 here
    assert gibbons_are_unif()["sign_vs_t"] == pytest.approx(1 / 3)
