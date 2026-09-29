"""Test hjorth_params."""

from morie.fn import _array_core as np
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

    def test_hjorth_parameters_recomputed(self):
        import math

        import pytest

        x = [0.5, 1.2, -0.3, 0.8, 2.0, -1.1, 0.4]

        def v0(a):
            m = sum(a) / len(a)
            return sum((t - m) ** 2 for t in a) / len(a)

        d1 = [x[i + 1] - x[i] for i in range(6)]
        d2 = [d1[i + 1] - d1[i] for i in range(5)]
        mob = math.sqrt(v0(d1) / v0(x))
        comp = math.sqrt(v0(d2) / v0(d1)) / mob
        r = hjorth_params(x)
        assert r.extra["activity"] == pytest.approx(v0(x), rel=1e-13)
        assert r.extra["mobility"] == pytest.approx(mob, rel=1e-13)
        assert r.extra["complexity"] == pytest.approx(comp, rel=1e-12)
