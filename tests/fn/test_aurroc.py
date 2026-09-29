"""Tests for morie.fn.aurroc: AUC as the normalised Mann-Whitney count."""

import math

from morie.fn.aurroc import aurroc


def test_mann_whitney():
    y = [int(math.sin(3.1 * k) > 0) for k in range(30)]
    s = [round(math.cos(1.3 * k) + 0.5 * y[k], 1) for k in range(30)]
    pos = [b for a, b in zip(y, s) if a]
    neg = [b for a, b in zip(y, s) if not a]
    ref = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    assert abs(float(aurroc(y, s)) - ref) < 1e-15
    assert float(aurroc([0, 1], [0.2, 0.9])) == 1.0
    assert float(aurroc([0, 1], [0.9, 0.2])) == 0.0
