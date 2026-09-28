"""Tests for morie.fn.zeklf -- Kulldorff spatial scan statistic (re-export of scanstat.kulldorff_scan)."""

import math

import pytest

from morie.fn.scanstat import kulldorff_scan as real
from morie.fn.zeklf import kull, kulldorff_scan, kulldorffscan


def test_zeklf_is_the_real_scan():
    assert kulldorff_scan is real and kull is real and kulldorffscan is real


def test_zeklf_most_likely_cluster_by_hand():
    P = [(float(i), 0.0) for i in range(6)]
    r = kulldorff_scan(P, [9, 8, 1, 1, 1, 0], [10] * 6, nsim=0)
    assert r.all_zones[0] == [0, 1]
    # 17 log(17 / (20/6 * 2)) + 3 log(3 / (20 - 20/3))
    want = 17 * math.log(17 / (40 / 6)) + 3 * math.log(3 / (20 - 40 / 6))
    assert r.all_tobs[0] == pytest.approx(want, abs=1e-12)
