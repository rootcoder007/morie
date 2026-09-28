"""Tests for morie.fn.zekls -- circular scan statistic (re-export of scanstat.kulldorff_scan)."""

from morie.fn.scanstat import kulldorff_scan
from morie.fn.zekls import scan, scan_circular, scancircular


def test_zekls_is_the_real_circular_scan():
    assert scan_circular is kulldorff_scan and scan is kulldorff_scan and scancircular is kulldorff_scan


def test_zekls_binomial_cluster():
    P = [(float(i), 0.0) for i in range(6)]
    r = scan_circular(P, [9, 8, 1, 1, 1, 0], [20] * 6, kind="binomial", nsim=0)
    assert r.all_zones[0] == [0, 1]
    assert r.all_tobs[0] > 0
