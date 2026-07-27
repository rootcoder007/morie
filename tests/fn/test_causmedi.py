"""Tests for causmedi.causal_mediation_imai."""

import numpy as np
import pytest

from morie.fn.causmedi import causal_mediation_imai


def _simple(seed=42, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_causmedi_basic():
    out = causal_mediation_imai(*_simple(), n_boot=200, seed=0)
    assert out["acme"] == pytest.approx(1.2, abs=0.15)
    assert out["ade"] == pytest.approx(0.7, abs=0.15)
    assert out["total"] == pytest.approx(out["acme"] + out["ade"])


def test_causmedi_edge():
    with pytest.raises(ValueError):
        causal_mediation_imai(*_simple(), n_boot=0)
    with pytest.raises(ValueError):
        causal_mediation_imai(*_simple(), alpha=1.5)
