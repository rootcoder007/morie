"""Tests for morie.fn.jamste -- James-Stein shrinkage."""

from morie.fn._containers import DescriptiveResult
from morie.fn.jamste import james_stein, jamste


class TestJamste:
    def test_alias(self):
        assert jamste is james_stein

    def test_shrinks(self):
        x = [10.0, -5.0, 3.0, 0.1, -2.0]
        result = james_stein(x)
        assert isinstance(result, DescriptiveResult)
        assert 0 <= result.value <= 1

    def test_needs_three(self):
        import pytest

        with pytest.raises(ValueError, match="requires >= 3"):
            james_stein([1.0, 2.0])


def test_grand_mean_target_uses_p_minus_3():
    """Efron-Morris: estimating the target costs a degree of freedom."""
    import pytest

    x = [10.0, -5.0, 3.0, 0.1, -2.0, 4.4]
    p = len(x)
    m = sum(x) / p
    ss = sum((v - m) ** 2 for v in x)
    c = max(0.0, 1 - (p - 3) * 2.5 / ss)
    r = james_stein(x, sigma2=2.5)
    assert r.value == pytest.approx(c, rel=1e-12)
    assert r.extra["js_estimates"] == pytest.approx([m + c * (v - m) for v in x], rel=1e-12)


def test_fixed_target_uses_p_minus_2_and_positive_part():
    import pytest

    x = [0.6, 0.4, -0.5, 0.3]
    ss = sum((v - 0.0) ** 2 for v in x)
    r = james_stein(x, target=0.0)
    assert r.value == pytest.approx(max(0.0, 1 - 2 / ss), rel=1e-12)
    assert r.value == 0.0  # ss < 2, so the positive part clips to full shrinkage
    assert r.extra["js_estimates"] == pytest.approx([0.0] * 4, abs=1e-15)
