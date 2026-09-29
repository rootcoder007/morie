"""Tests for morie.fn.prdur: values recomputed from the definition."""

import math

from morie.fn.prdur import pr_duration


def test_pr_intervals_in_seconds():
    p_on, q_on, fs = [100, 900, 1700], [260, 1070, 1850], 500.0
    r = pr_duration(p_on, q_on, fs=fs)
    pr = [(b - a) / fs for a, b in zip(p_on, q_on)]
    assert max(abs(a - b) for a, b in zip(r.extra["pr_intervals"], pr)) < 1e-15
    m = sum(pr) / 3
    assert abs(r.value - m) < 1e-15
    assert abs(r.extra["std_pr"] - math.sqrt(sum((v - m) ** 2 for v in pr) / 2)) < 1e-15
