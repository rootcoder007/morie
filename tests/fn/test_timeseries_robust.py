"""Unit-root testing and robust regression, anchored on R.

References are ``urca::ur.df`` and ``MASS::rlm`` run on the same
fixtures: a persistent AR(1) for the unit-root test, and a regression
with two deliberately gross outliers for the robust fit.
"""
import pytest

from morie.fn import _robust_core as rb

N = 60
Z = [((i * 13) % 17) / 17 - 0.5 for i in range(N)]
Y = [0.0] * N
Y[0] = Z[0]
for _t in range(1, N):
    Y[_t] = 0.85 * Y[_t - 1] + Z[_t]

M = 30
RX1 = [((i * 7) % 13) + 0.5 * ((i * 3) % 5) for i in range(M)]
RX2 = [((i * 5) % 11) - 0.25 * ((i * 2) % 7) for i in range(M)]
RE = [((i * 11) % 19) / 19 - 0.5 for i in range(M)]
YR = [2 + 1.4 * RX1[i] - 0.7 * RX2[i] + RE[i] for i in range(M)]
YR[6] += 25.0
YR[21] -= 30.0
RX = [[RX1[i], RX2[i]] for i in range(M)]


def test_adf_matches_urca_ur_df():
    want = {"none": -3.59709722078672, "drift": -5.81354784454201,
            "trend": -5.76130487932429}
    for kind, target in want.items():
        got = rb.adf_test(Y, lags=1, kind=kind)["statistic"]
        assert abs(got - target) < 1e-12, kind


def test_adf_rejects_a_unit_root_for_a_stationary_series():
    # generated with phi = 0.85, so stationary: reject at 5 per cent
    r = rb.adf_test(Y, lags=1, kind="drift")
    assert r["reject_5pct"] is True
    assert r["statistic"] < r["critical_values"]["5pct"]


def test_adf_does_not_reject_for_a_random_walk():
    rw = [0.0] * N
    for t in range(1, N):
        rw[t] = rw[t - 1] + Z[t]
    r = rb.adf_test(rw, lags=1, kind="drift")
    assert r["reject_5pct"] is False


def test_adf_critical_values_are_ordered_and_negative():
    c = rb.adf_test(Y, lags=1, kind="trend")["critical_values"]
    assert c["1pct"] < c["5pct"] < c["10pct"] < 0


def test_adf_validates_its_arguments():
    with pytest.raises(ValueError):
        rb.adf_test(Y, kind="quadratic")
    with pytest.raises(ValueError):
        rb.adf_test([1.0, 2.0], lags=5)


def test_rlm_matches_mass_rlm():
    r = rb.rlm(YR, RX)
    for got, want in zip(r["coef"], (1.67875433365285, 1.42945187591586,
                                     -0.678206572264915)):
        assert abs(got - want) < 1e-4
    assert abs(r["scale"] - 0.348036179139661) < 1e-3


def test_rlm_downweights_exactly_the_planted_outliers():
    r = rb.rlm(YR, RX)
    w = r["weights"]
    assert w[6] < 0.05 and w[21] < 0.05        # the two gross ones
    clean = [w[i] for i in range(M) if i not in (6, 21)]
    assert min(clean) > 0.5
    # R gives 0.0187 and 0.0154 for these two
    assert abs(w[6] - 0.0187128301143033) < 1e-3
    assert abs(w[21] - 0.0154422749052354) < 1e-3


def test_rlm_recovers_the_clean_fit_better_than_least_squares():
    # The right benchmark is the CLEAN-data fit, not the generating
    # parameter: with n = 30 and errors up to 0.5, even an uncontaminated
    # fit misses 1.4 by a couple of hundredths, so comparing to the truth
    # measures noise rather than robustness.  (The planted +25 and -30
    # also partly cancel, so the OLS *slope* here is barely disturbed --
    # it is the whole coefficient vector that moves.)
    from morie.fn import _regression_core as rg
    clean = [2 + 1.4 * RX1[i] - 0.7 * RX2[i] + RE[i] for i in range(M)]
    target = rg.ols(clean, RX)["coef"]
    ols = rg.ols(YR, RX)["coef"]
    rob = rb.rlm(YR, RX)["coef"]

    def dist(a):
        return max(abs(a[j] - target[j]) for j in range(len(target)))

    assert dist(rob) < dist(ols)
    assert dist(rob) < 0.05


def test_rlm_reduces_to_least_squares_without_outliers():
    clean = [2 + 1.4 * RX1[i] - 0.7 * RX2[i] + RE[i] for i in range(M)]
    from morie.fn import _regression_core as rg
    a = rb.rlm(clean, RX)["coef"]
    b = rg.ols(clean, RX)["coef"]
    for x, y in zip(a, b):
        assert abs(x - y) < 0.05


def test_rlm_uses_mad_about_zero_like_mass():
    # MASS::rlm computes mad(resid, 0); centring on the residual median
    # instead shifts the scale and every weight with it
    r = rb.rlm(YR, RX)
    resid = sorted(abs(t) for t in r["residuals"])
    n = len(resid)
    mad0 = (resid[n // 2] if n % 2
            else 0.5 * (resid[n // 2 - 1] + resid[n // 2]))
    assert abs(r["scale"] - mad0 / 0.6745) < 1e-12


def test_rlm_validates_shapes():
    with pytest.raises(ValueError):
        rb.rlm(YR[:5], RX)
