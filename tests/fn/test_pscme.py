"""Tests for pscme.path_specific_causal_effect."""

import numpy as np
import pytest

from morie.fn.pscme import path_specific_causal_effect


def test_pscme_basic():
    rng = np.random.default_rng(42)
    n = 3000
    x = rng.normal(size=n)
    m1 = 0.6 * x + rng.normal(scale=0.6, size=n)
    m2 = 0.4 * x + 0.5 * m1 + rng.normal(scale=0.6, size=n)
    y = 0.3 * x + 0.7 * m1 + 0.9 * m2 + rng.normal(scale=0.6, size=n)
    out = path_specific_causal_effect(x, m1, m2, y)
    assert set(out["paths"]) == {"X->Y", "X->M1->Y", "X->M2->Y", "X->M1->M2->Y"}
    assert out["paths"]["X->M1->M2->Y"] == pytest.approx(0.6 * 0.5 * 0.9, abs=0.06)
    assert out["total"] == pytest.approx(sum(out["paths"].values()))


def test_pscme_edge():
    with pytest.raises(ValueError):
        path_specific_causal_effect([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5)
