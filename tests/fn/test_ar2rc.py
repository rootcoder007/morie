"""Test ar_to_reflection (ar2rc)."""

from morie.fn._containers import DescriptiveResult
from morie.fn.ar2rc import ar2rc, ar_to_reflection


class TestAr2rc:
    def test_basic(self):
        result = ar_to_reflection([1.0, -0.5, 0.2])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ar_to_reflection"
        rc = result.extra["rc"]
        assert len(rc) == 2

    def test_alias(self):
        assert ar2rc is ar_to_reflection

    def test_second_order_step_down_recomputed(self):
        """Step-down (Levinson) recursion for AR(2): k2 = a2, k1 = a1 / (1 + a2)."""
        import pytest

        a1, a2 = -0.5, 0.2
        r = ar_to_reflection([1.0, a1, a2])
        assert [float(v) for v in r.extra["rc"]] == pytest.approx([a1 / (1 + a2), a2], rel=1e-14)
        assert r.value == pytest.approx(a1 / (1 + a2), rel=1e-14)
