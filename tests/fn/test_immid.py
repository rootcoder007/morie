"""Tests for immid.index_moderated_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.immid import index_moderated_mediation


def test_immid_basic():
    rng = np.random.default_rng(42)
    n = 4000
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    m = 0.3 * x + 0.2 * w + 0.6 * x * w + rng.normal(scale=0.6, size=n)
    y = 0.2 * x + 1.2 * m + rng.normal(scale=0.6, size=n)
    out = index_moderated_mediation(x, m, y, w, w_values=[-1.0, 0.0, 1.0])
    assert out["index"] == pytest.approx(0.6 * 1.2, abs=0.1)
    assert np.diff(out["conditional_indirect"]) == pytest.approx([out["index"]] * 2)


def test_immid_edge():
    with pytest.raises(ValueError):
        index_moderated_mediation([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5)
