"""Tests for seqM.sequential_mediators."""

import numpy as np
import pytest

from morie.fn.medstg import sequential_mediation
from morie.fn.seqM import sequential_mediators


def test_seqM_basic():
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(size=n)
    m1 = 0.6 * x + rng.normal(scale=0.6, size=n)
    m2 = 0.4 * x + 0.5 * m1 + rng.normal(scale=0.6, size=n)
    y = 0.3 * x + 0.7 * m1 + 0.9 * m2 + rng.normal(scale=0.6, size=n)
    a = sequential_mediators(y, x, m1, m2)
    b = sequential_mediation(x, m1, m2, y)
    assert a["serial"] == pytest.approx(b["serial"])


def test_seqM_edge():
    with pytest.raises(ValueError):
        sequential_mediators([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5)  # too few obs
