"""Tests for rghrv.rangayyan_hrv.

Spec: Task Force of the ESC/NASPE (1996), Circulation 93(5):1043-1065 --
the definitions of SDNN, RMSSD and pNN50. Expected values are computed by
hand from those definitions on a short RR series.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rghrv import rangayyan_hrv


# RR = [800, 810, 790, 805, 795] ms
#   mean            = 800.0
#   SDNN (ddof=1)   : deviations 0, 10, -10, 5, -5 -> sum sq = 250
#                     sqrt(250/4) = sqrt(62.5)
#   successive diffs: 10, -20, 15, -10 -> sum sq = 100+400+225+100 = 825
#   RMSSD           = sqrt(825/4) = sqrt(206.25)
#   pNN50           : |diff| > 50 -> none -> 0.0 %
RR = np.array([800.0, 810.0, 790.0, 805.0, 795.0])


def test_rghrv_matches_task_force_definitions():
    r = rangayyan_hrv(RR)
    assert r["meanNN"] == pytest.approx(800.0)
    assert r["SDNN"] == pytest.approx(np.sqrt(62.5))
    assert r["RMSSD"] == pytest.approx(np.sqrt(206.25))
    assert r["pNN50"] == pytest.approx(0.0)
    assert r["n"] == 5


def test_rghrv_heart_rate_from_mean_nn():
    # 800 ms mean NN -> 60000/800 = 75 bpm exactly
    assert rangayyan_hrv(RR)["heart_rate_bpm"] == pytest.approx(75.0)


def test_rghrv_pnn50_counts_only_differences_over_50ms():
    # diffs: 100, -100, 10 -> two of three exceed 50 ms -> 200/3 %
    rr = np.array([800.0, 900.0, 800.0, 810.0])
    assert rangayyan_hrv(rr)["pNN50"] == pytest.approx(200.0 / 3.0)


def test_rghrv_constant_rr_has_zero_variability():
    r = rangayyan_hrv(np.full(10, 850.0))
    assert r["SDNN"] == pytest.approx(0.0)
    assert r["RMSSD"] == pytest.approx(0.0)
    assert r["pNN50"] == pytest.approx(0.0)


def test_rghrv_needs_at_least_two_intervals():
    # SDNN with ddof=1 and RMSSD both require >= 2 intervals; a single beat
    # carries no variability information at all.
    with pytest.raises(ValueError, match="at least 2"):
        rangayyan_hrv(np.array([800.0]))
