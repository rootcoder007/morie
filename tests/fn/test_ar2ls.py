"""Test ar_to_lsf (ar2ls)."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.ar2ls import ar2ls, ar_to_lsf


class TestAr2ls:
    def test_basic(self):
        result = ar_to_lsf([1.0, -0.5, 0.2])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ar_to_lsf"
        lsf = result.extra["lsf"]
        assert len(lsf) > 0

    def test_lsf_in_range(self):
        result = ar_to_lsf([1.0, -0.8, 0.3])
        lsf = result.extra["lsf"]
        for w in lsf:
            assert 0 < w < np.pi

    def test_alias(self):
        assert ar2ls is ar_to_lsf


def test_ar2ls_second_order_lsf_closed_form():
    """AR(2): P(z) = (1 + z)(z^2 + (a1 + a2 - 1) z + 1) and
    Q(z) = (z - 1)(z^2 + (a1 - a2 + 1) z + 1), so the two LSFs are
    arccos(-(a1 + a2 - 1)/2) and arccos(-(a1 - a2 + 1)/2)."""
    import math

    import pytest

    a1, a2 = -0.9, 0.4
    wp = math.acos(-(a1 + a2 - 1) / 2)
    wq = math.acos(-(a1 - a2 + 1) / 2)
    r = ar_to_lsf([1.0, a1, a2])
    assert [float(v) for v in r.extra["lsf"]] == pytest.approx(sorted([wp, wq]), rel=1e-9)
    assert r.value == pytest.approx(min(wp, wq), rel=1e-9)
