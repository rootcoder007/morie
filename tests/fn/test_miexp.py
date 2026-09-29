"""Tests for morie.fn.miexp."""

import pytest

from morie.fn.miexp import miexp


def test_expectation_formula():
    for n in (2, 5, 37):
        assert abs(miexp(n).statistic - (-1.0 / (n - 1))) < 1e-15


def test_small_n_rejected():
    with pytest.raises(ValueError):
        miexp(1)
