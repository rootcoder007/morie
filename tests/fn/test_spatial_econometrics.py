"""Spatial autocorrelation and spatial-econometric estimators.

Anchored on the R reference implementations -- spdep::Szero,
spdep::moran.test, spatialreg::stsls, spatialreg::GMerrorsar -- run on
the same 5x5 rook lattice.  These are the packages the Bivand,
Pebesma & Gomez-Rubio examples actually use, so matching them is what
makes the capability real rather than merely present.
"""
import math

import pytest

from morie.fn import _robust_core as rb


def rook_lattice(k=5):
    """Row-standardised rook contiguity on a k x k grid."""
    n = k * k

    def idx(r, c):
        return r * k + c

    W = [[0.0] * n for _ in range(n)]
    for r in range(k):
        for c in range(k):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < k and 0 <= cc < k:
                    W[idx(r, c)][idx(rr, cc)] = 1.0
    for i in range(n):
        s = sum(W[i])
        if s:
            W[i] = [v / s for v in W[i]]
    return W


W25 = rook_lattice(5)
X25 = [float((i * 7) % 11) + 0.5 * ((i * 3) % 5) for i in range(25)]


def test_weights_totals_match_spdep_szero():
    t = rb.weights_totals(W25)
    assert abs(t["S0"] - 25.0) < 1e-12
    assert abs(t["S1"] - 16.194444444444) < 1e-11
    assert abs(t["S2"] - 100.666666666667) < 1e-11
    # S0 of a row-standardised matrix is n by construction
    assert abs(t["S0"] - len(W25)) < 1e-12


def test_morans_i_matches_spdep():
    assert abs(rb.morans_i(X25, W25) - (-0.153400121433)) < 1e-11


def test_morans_i_test_randomisation_matches_spdep():
    r = rb.morans_i_test(X25, W25, randomisation=True)
    assert abs(r["estimate"] - (-0.153400121433)) < 1e-11
    assert abs(r["expectation"] - (-0.041666666667)) < 1e-11
    assert abs(r["variance"] - 0.023435730140) < 1e-11
    assert abs(r["statistic"] - (-0.729867428466)) < 1e-11
    assert abs(r["p_value"] - 0.767264388244) < 1e-11


def test_morans_i_test_normality_matches_spdep():
    r = rb.morans_i_test(X25, W25, randomisation=False)
    assert abs(r["variance"] - 0.022571225071) < 1e-11
    assert abs(r["statistic"] - (-0.743713494173)) < 1e-11
    assert abs(r["p_value"] - 0.771475088018) < 1e-11


def test_moran_expectation_is_minus_one_over_n_minus_one():
    for k in (4, 5, 6):
        W = rook_lattice(k)
        x = [float((i * 3) % 7) for i in range(k * k)]
        r = rb.morans_i_test(x, W)
        assert abs(r["expectation"] + 1.0 / (k * k - 1)) < 1e-12


def test_randomisation_and_normality_differ_on_skewed_data():
    # the two nulls only coincide when the kurtosis is that of a
    # normal; on skewed data they must not agree
    skew = [float(i) ** 3 for i in range(25)]
    a = rb.morans_i_test(skew, W25, randomisation=True)["variance"]
    b = rb.morans_i_test(skew, W25, randomisation=False)["variance"]
    assert abs(a - b) > 1e-6


def _sar_fixture():
    n = 25
    x1 = [((i * 7) % 11) + 0.5 * ((i * 3) % 5) for i in range(n)]
    x2 = [((i * 5) % 7) - 0.25 * ((i * 2) % 3) for i in range(n)]
    e = [((i * 13) % 17) / 17 - 0.5 for i in range(n)]
    rhs = [2 + 1.5 * x1[i] - 0.8 * x2[i] + e[i] for i in range(n)]
    A = [[(1.0 if i == j else 0.0) - 0.4 * W25[i][j] for j in range(n)]
         for i in range(n)]
    y = rb._solve_local(A, rhs)
    return y, [[x1[i], x2[i]] for i in range(n)]


def test_spatial_2sls_matches_spatialreg_stsls():
    y, X = _sar_fixture()
    s = rb.spatial_2sls(y, X, W25)
    assert abs(s["rho"] - 0.392173355844) < 1e-11
    assert abs(s["beta"][0] - 1.930187859839) < 1e-11
    assert abs(s["beta"][1] - 1.506892839923) < 1e-11
    assert abs(s["beta"][2] - (-0.762612153824)) < 1e-11


def test_spatial_2sls_recovers_the_generating_rho():
    y, X = _sar_fixture()
    # the fixture was built with rho = 0.4
    assert abs(rb.spatial_2sls(y, X, W25)["rho"] - 0.4) < 0.05


def test_gm_error_sar_matches_spatialreg_gmerrorsar():
    y, X = _sar_fixture()
    g = rb.gm_error_sar(y, X, W25)
    # spatialreg::GMerrorsar gives lambda 0.513267505390; the residual
    # 2e-9 is nlminb's stopping tolerance against our golden-section
    # search -- both sit at the same optimum, criterion 2.686596e-03
    assert abs(g["lambda"] - 0.513267505390) < 1e-8
    assert abs(g["beta"][0] - 7.630934971425) < 1e-8
    assert abs(g["beta"][1] - 1.378029513797) < 1e-8
    assert abs(g["beta"][2] - (-0.495944119576)) < 1e-8


def test_gm_criterion_is_at_least_as_good_as_r_optimum():
    # guard against "close but worse": our lambda must not have a
    # higher moment criterion than R's
    y, X = _sar_fixture()
    g = rb.gm_error_sar(y, X, W25)
    assert g["criterion"] <= 2.686597e-03


def test_spatial_estimators_reject_bad_shapes():
    y, X = _sar_fixture()
    with pytest.raises(Exception):
        rb.spatial_2sls(y[:5], X, W25)
    with pytest.raises(Exception):
        rb.morans_i([1.0, 2.0], W25)
