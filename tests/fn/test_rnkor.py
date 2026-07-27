"""Tests for rnkor.rank_order_statistics."""

import numpy as np
import pytest

from morie.fn.rnkor import rank_order_statistics


def test_rnkor_signed_rank_sums_partition_n_n_plus_1_over_2():
    """Without ties or zeros, W+ + W- = n(n+1)/2 -- an identity."""
    x = np.array([1.2, -0.5, 2.3, -3.1, 0.7, 1.9])
    r = rank_order_statistics(x, mu0=0.0)
    n = 6
    assert float(r["W_plus"]) + float(r["W_minus"]) == pytest.approx(n * (n + 1) / 2, abs=1e-12)


def test_rnkor_hand_computed_case():
    """x - mu0 = (1, -2, 3): |d| ranks are (1, 2, 3), so W+ = 1 + 3 = 4
    and W- = 2."""
    r = rank_order_statistics(np.array([1.0, -2.0, 3.0]), mu0=0.0)
    assert float(r["W_plus"]) == pytest.approx(4.0, abs=1e-12)
    assert float(r["W_minus"]) == pytest.approx(2.0, abs=1e-12)


def test_rnkor_zeros_are_dropped():
    r = rank_order_statistics(np.array([0.0, 1.0, -1.0, 2.0]), mu0=0.0)
    assert int(r["n_nonzero"]) == 3
