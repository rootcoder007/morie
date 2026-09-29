"""Tests for morie.fn.bsastat: Pearson correlation recomputed."""

import math

import pytest

from morie.fn.bsastat import corrcoef


def test_pearson_r_and_raw_cosine():
    x = [1.0, 2.0, 4.0, 3.0, 6.0]
    y = [2.1, 2.9, 5.2, 3.8, 6.9]
    mx, my = sum(x) / 5, sum(y) / 5
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    cos = sum(a * b for a, b in zip(x, y)) / math.sqrt(sum(a * a for a in x) * sum(b * b for b in y))
    out = corrcoef(x, y)
    assert abs(out["r"] - r) < 1e-14
    assert abs(out["cosine_without_removing_means"] - cos) < 1e-14
    with pytest.raises(ValueError):
        corrcoef([1.0, 1.0], [2.0, 3.0])
