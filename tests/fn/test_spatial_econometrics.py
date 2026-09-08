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


# --- the six formerly named-but-template modules ---------------------
def test_spatial_lag_model_matches_spatialreg_lagsarlm():
    y, X = _sar_fixture()
    m = rb.spatial_lag_model(y, X, W25)
    assert abs(m["rho"] - 0.391406232872727) < 1e-7
    assert abs(m["beta"][0] - 1.94007428339988) < 1e-6
    assert abs(m["beta"][1] - 1.50683946841365) < 1e-7
    assert abs(m["beta"][2] - (-0.762066735722032)) < 1e-7
    assert abs(m["sigma2"] - 0.0785475951546229) < 1e-9


def test_spatial_error_model_matches_spatialreg_errorsarlm():
    y, X = _sar_fixture()
    m = rb.spatial_error_model(y, X, W25)
    assert abs(m["lambda"] - 0.824444050787294) < 1e-6
    assert abs(m["beta"][0] - 7.94062292250759) < 1e-6
    assert abs(m["beta"][1] - 1.34203947781064) < 1e-7
    assert abs(m["beta"][2] - (-0.508793220452759)) < 1e-7
    assert abs(m["sigma2"] - 0.437341709973207) < 1e-7


def test_lag_model_beats_ols_when_rho_is_real():
    # the fixture really was generated with rho = 0.4, so ML should
    # find it and OLS on y ~ X should be biased
    y, X = _sar_fixture()
    m = rb.spatial_lag_model(y, X, W25)
    assert abs(m["rho"] - 0.4) < 0.05
    # and 2SLS, which targets the same parameter, should agree closely
    s = rb.spatial_2sls(y, X, W25)
    assert abs(m["rho"] - s["rho"]) < 0.01


def test_logdet_matches_a_direct_determinant():
    # log|I - rho W| on a small matrix, checked against the 2x2 formula
    W = [[0.0, 0.5], [0.5, 0.0]]
    for rho in (0.0, 0.3, -0.4):
        want = math.log(abs(1 - rho * rho * 0.25))
        assert abs(rb._logdet_I_minus(rho, W) - want) < 1e-12
    assert rb._logdet_I_minus(0.0, W) == 0.0


def test_ripley_k_recovers_pi_r_squared_under_csr():
    # on a regular grid with edge correction K(r) should track the CSR
    # benchmark pi r^2 reasonably over mid-range r
    pts = [[float(i), float(j)] for i in range(10) for j in range(10)]
    r = [1.5, 2.0, 2.5]
    out = rb.ripley_k(pts, r)
    for k, rr in zip(out["K"], r):
        assert abs(k - math.pi * rr * rr) / (math.pi * rr * rr) < 0.35
    # K is non-decreasing in r
    assert all(a <= b + 1e-9
               for a, b in zip(out["K"], out["K"][1:]))


def test_ripley_k_edge_correction_raises_k_near_the_boundary():
    pts = [[float(i), float(j)] for i in range(6) for j in range(6)]
    with_c = rb.ripley_k(pts, [2.0], edge_correction=True)["K"][0]
    without = rb.ripley_k(pts, [2.0], edge_correction=False)["K"][0]
    # uncorrected K misses neighbours outside the window, so it is lower
    assert without < with_c


def test_ripley_l_linearises_k():
    pts = [[float(i), float(j)] for i in range(8) for j in range(8)]
    out = rb.ripley_k(pts, [1.0, 2.0, 3.0])
    for k, l in zip(out["K"], out["L"]):
        assert abs(l - math.sqrt(k / math.pi)) < 1e-12


def test_ripley_k_rejects_bad_input():
    with pytest.raises(ValueError):
        rb.ripley_k([[0.0, 0.0]], [1.0])
    with pytest.raises(ValueError):
        rb.ripley_k([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [1.0])


def test_cokriging_reduces_to_ordinary_kriging_without_cross_structure():
    # with a zero cross-variogram the covariate carries no information,
    # so the mu weights must vanish and the lambdas must still sum to 1
    pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    z1 = [1.0, 2.0, 3.0, 4.0]
    z2 = [9.0, 9.0, 9.0, 9.0]
    out = rb.cokriging(pts, z1, z2, [0.5, 0.5],
                       cross_vario=lambda h: 0.0)
    assert abs(sum(out["lambda"]) - 1.0) < 1e-9
    assert max(abs(m) for m in out["mu"]) < 1e-9
    # symmetric configuration and target -> prediction is the mean
    assert abs(out["prediction"] - 2.5) < 1e-6


def test_cokriging_weights_satisfy_the_unbiasedness_constraints():
    pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]
    z1 = [1.0, 2.0, 3.0, 7.0]
    z2 = [2.0, 1.0, 4.0, 5.0]
    out = rb.cokriging(pts, z1, z2, [0.7, 0.4])
    assert abs(sum(out["lambda"]) - 1.0) < 1e-9
    assert abs(sum(out["mu"])) < 1e-9


def test_cokriging_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rb.cokriging([[0.0, 0.0], [1.0, 1.0]], [1.0], [1.0, 2.0],
                     [0.5, 0.5])


def test_randomised_response_transition_probabilities():
    # P(keep) / P(flip) must equal e^eps -- that IS the privacy claim
    for eps in (0.5, 1.0, 2.0):
        for k in (2, 4, 10):
            r = rb.local_dp_randomised_response([0] * 5, k, eps)
            assert abs(r["p_keep"] / r["p_flip"] - math.exp(eps)) < 1e-12
            assert abs(r["p_keep"] + (k - 1) * r["p_flip"] - 1.0) < 1e-12


def test_randomised_response_debiased_estimate_is_unbiased():
    # a large sample from a known distribution must be recovered
    truth = [0] * 6000 + [1] * 3000 + [2] * 1000
    r = rb.local_dp_randomised_response(truth, 3, 2.0, seed=11)
    for got, want in zip(r["estimate"], (0.6, 0.3, 0.1)):
        assert abs(got - want) < 0.03
    assert abs(sum(r["estimate"]) - 1.0) < 1e-9


def test_randomised_response_is_noisier_at_smaller_epsilon():
    truth = [0] * 2000 + [1] * 2000
    tight = rb.local_dp_randomised_response(truth, 2, 3.0, seed=5)
    loose = rb.local_dp_randomised_response(truth, 2, 0.2, seed=5)
    # stronger privacy (smaller eps) keeps fewer true values
    assert loose["p_keep"] < tight["p_keep"]


def test_randomised_response_validates_its_arguments():
    with pytest.raises(ValueError):
        rb.local_dp_randomised_response([0, 1], 1, 1.0)
    with pytest.raises(ValueError):
        rb.local_dp_randomised_response([0, 1], 2, 0.0)
    with pytest.raises(ValueError):
        rb.local_dp_randomised_response([0, 5], 2, 1.0)
