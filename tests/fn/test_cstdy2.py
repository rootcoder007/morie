"""Tests for cstdy2.custody_days_credit."""

import pytest

from morie.fn.cstdy2 import custody_days_credit


def test_credit():
    d = [10, 30, 45, 0, 7]
    r = custody_days_credit(d)
    assert r.extra["credited"] == [1.5 * v for v in d]
    assert r.estimate == sum(1.5 * v for v in d) / 5
    assert custody_days_credit(d, credit_ratio=1.0).extra["total_credited"] == 92.0
    with pytest.raises(ValueError):
        custody_days_credit(d, credit_ratio=2.0)
