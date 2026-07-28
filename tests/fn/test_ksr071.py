"""Tests for ksr071 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr071 import kosorok_ch3_log_profile_expansion


def test_ksr071_basic():
    out = kosorok_ch3_log_profile_expansion([0.5], [0.4], [[4.0]], n=100)
    assert out["quadratic_term"] == pytest.approx(0.5 * 100 * 0.01 * 4.0)


def test_ksr071_edge():
    with pytest.raises(ValueError):
        kosorok_ch3_log_profile_expansion([0.5], [0.4], [[4.0]])  # n missing
