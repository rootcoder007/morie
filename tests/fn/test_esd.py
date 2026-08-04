"""Tests for esd.generalized_esd (Rosner 1983 generalized ESD).

Anchored on the published worked example in the NIST/SEMATECH
e-Handbook, section 1.3.5.17.3, which uses Rosner's own 54-point data
set with r = 10 and alpha = 0.05 and reports three outliers,
R_1 = 3.118, R_3 = 3.179, lambda_1 = 3.158, lambda_10 = 3.085.  The
handbook truncates to three decimals, hence the 2e-3 tolerance.
"""

import pytest

from morie.fn.esd import generalized_esd

ROSNER = [
    -0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49,
    1.55, 1.56, 1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96,
    1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40,
    2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59,
    3.68, 4.30, 4.64, 5.34, 5.42, 6.01,
]


def test_esd_matches_nist_worked_example():
    res = generalized_esd(ROSNER, 0.05, 10)
    assert res["n_outliers"] == 3
    assert res["R"][0] == pytest.approx(3.118, abs=2e-3)
    assert res["R"][2] == pytest.approx(3.179, abs=2e-3)
    assert res["lam"][0] == pytest.approx(3.158, abs=2e-3)
    assert res["lam"][9] == pytest.approx(3.085, abs=2e-3)
    # the three flagged points are the three largest values
    flagged = sorted(ROSNER[i] for i in res["outlier_index"])
    assert flagged == [5.34, 5.42, 6.01]


def test_esd_clean_sample_flags_nothing():
    x = [float(i) for i in range(30)]
    assert generalized_esd(x, 0.05, 3)["n_outliers"] == 0


def test_esd_rejects_bad_r():
    with pytest.raises(ValueError):
        generalized_esd([1.0, 2.0, 3.0], 0.05, 5)
