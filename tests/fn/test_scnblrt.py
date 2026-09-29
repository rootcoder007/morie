"""Tests for morie.fn.scnblrt: boundary-corrected likelihood-ratio p-value."""

import math

from morie.fn.scnblrt import scnblrt


def test_chi_bar_square_pvalue():
    r = scnblrt(-40.0, -42.3)
    lr = 2 * 2.3
    assert abs(r.statistic - lr) < 1e-13
    assert abs(r.p_value - 0.5 * math.erfc(math.sqrt(lr / 2))) < 1e-13


def test_negative_lr_truncated_and_df2():
    assert scnblrt(-43.0, -42.0).statistic == 0.0
    assert abs(scnblrt(-40.0, -42.0, df=2).p_value - math.exp(-2.0)) < 1e-12
