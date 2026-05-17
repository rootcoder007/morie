"""Tests for morie.fn.infgai -- information gain."""

import numpy as np
import pandas as pd
from morie.fn.infgai import information_gain, infgai
from morie.fn._containers import DescriptiveResult


class TestInfgai:
    def test_alias(self):
        assert infgai is information_gain

    def test_perfect_predictor(self):
        df = pd.DataFrame({"outcome": [0, 0, 1, 1], "x": [0, 0, 1, 1]})
        result = information_gain(df, target="outcome", feature="x")
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.5

    def test_random_feature(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"outcome": rng.integers(0, 2, 200), "x": rng.normal(0, 1, 200)})
        result = information_gain(df, target="outcome", feature="x")
        assert result.value < 0.5
