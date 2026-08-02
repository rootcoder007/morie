"""Test form_factor_fn."""

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.frmfc import alias, form_factor_fn


class TestFormFactorFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = form_factor_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_positive(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = form_factor_fn(x)
        assert isinstance(result.value, float)
        assert result.value > 0.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = form_factor_fn(x)
        assert result.name == "form_factor"

    def test_alias(self):
        assert alias is form_factor_fn
