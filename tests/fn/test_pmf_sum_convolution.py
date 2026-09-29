"""Tests for morie.fn.pmf_sum_convolution: values recomputed from first principles."""

from morie.fn.pmf_sum_convolution import pmf_sum_convolution


def test_two_dice():
    d = [1, 2, 3, 4, 5, 6]
    r = pmf_sum_convolution(d, [1 / 6] * 6, d, [1 / 6] * 6)
    assert r["values"] == [float(s) for s in range(2, 13)]
    for s, p in zip(r["values"], r["probs"]):
        assert abs(p - (6 - abs(s - 7)) / 36) < 1e-15
