"""Tests for s2var (unbiased sample variance)."""

import pytest

from morie.fn.s2var import s2var


def test_s2var_is_the_n_minus_1_variance():
    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(x) / 5
    assert s2var(x) == pytest.approx(sum((v - m) ** 2 for v in x) / 4, rel=1e-14)
    with pytest.raises(ValueError):
        s2var([1.0])
