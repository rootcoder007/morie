"""Test hjorth_params."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.hjrth import alias, hjorth_params


class TestHjorthParams:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hjorth_params(x)
        assert isinstance(result, DescriptiveResult)

    def test_extra_has_activity(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hjorth_params(x)
        assert "activity" in result.extra

    def test_extra_has_mobility(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hjorth_params(x)
        assert "mobility" in result.extra

    def test_extra_has_complexity(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hjorth_params(x)
        assert "complexity" in result.extra

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hjorth_params(x)
        assert result.name == "hjorth"

    def test_alias(self):
        assert alias is hjorth_params
