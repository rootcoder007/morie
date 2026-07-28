"""Tests for ksr036 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr036 import kosorok_ch2_donsker_bracketing_theorem


def test_ksr036_basic():
    out = kosorok_ch2_donsker_bracketing_theorem(lambda e: (1 / e) ** 2)
    assert out["sufficient_condition_met"] is True


def test_ksr036_edge():
    # the key is named 'sufficient', because the theorem is not an iff
    out = kosorok_ch2_donsker_bracketing_theorem(lambda e: (1 / e) ** 2)
    assert "sufficient_condition_met" in out
