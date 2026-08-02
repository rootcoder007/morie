"""Tests for pluginM.plug_in_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.pluginM import plug_in_mediation


def test_pluginM_basic():
    rng = np.random.default_rng(42)
    n = 2000
    c = rng.normal(size=n)
    x = 0.5 * c + rng.normal(size=n)
    m = 0.8 * x + 0.6 * c + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + 0.5 * c + rng.normal(scale=0.7, size=n)
    result = plug_in_mediation(x, m, y, c=c)
    assert result["nie"] == pytest.approx(1.2, abs=0.15)  # 0.8 * 1.5
    assert result["nde"] == pytest.approx(0.7, abs=0.15)
    assert result["te"] == pytest.approx(result["nde"] + result["nie"])


def test_pluginM_edge():
    with pytest.raises(ValueError):
        plug_in_mediation([1.0, 2.0], [1.0], [1.0, 2.0])  # length mismatch
    with pytest.raises(ValueError):
        plug_in_mediation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])  # too few
