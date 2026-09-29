"""Test ergodicity_test (ergod)."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.ergod import ergod, ergodicity_test


class TestErgodicityTest:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        result = ergodicity_test(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ergodicity_test"

    def test_stationary_ergodic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        result = ergodicity_test(x, n_segments=10)
        assert result.extra["ergodic"] is True

    def test_too_short(self):
        with pytest.raises(ValueError):
            ergodicity_test([1.0, 2.0], n_segments=5)

    def test_alias(self):
        assert ergod is ergodicity_test


def test_segment_mean_variance_recomputed():
    x = [0.5, 0.9, 0.4, 1.3, 1.1, 0.2, -0.3, 0.1, 0.8, 0.6, 1.5, 1.0]
    segs = [x[i * 3 : (i + 1) * 3] for i in range(4)]
    sm = [sum(s) / 3 for s in segs]
    mm = sum(sm) / 4
    v = sum((m - mm) ** 2 for m in sm) / 4
    mx = sum(x) / 12
    vx = sum((t - mx) ** 2 for t in x) / 12
    r = ergodicity_test(x, n_segments=4)
    assert r.value == pytest.approx(v, rel=1e-13)
    assert r.extra["segment_means"] == pytest.approx(sm, rel=1e-14)
    assert r.extra["ergodic"] == (v < vx / 4 * 4)
