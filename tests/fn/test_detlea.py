"""Tests for morie.fn.detlea -- data leakage detection."""

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult
from morie.fn.detlea import detect_leakage, detlea


class TestDetlea:
    def test_alias(self):
        assert detlea is detect_leakage

    def test_detects_perfect_correlation(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 100).astype(float)
        df = pd.DataFrame({"outcome": y, "leaked": y, "noise": rng.normal(0, 1, 100)})
        result = detect_leakage(df, target="outcome")
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_no_leakage(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"outcome": rng.integers(0, 2, 50), "x": rng.normal(0, 10, 50)})
        result = detect_leakage(df, target="outcome")
        assert result.value == 0
