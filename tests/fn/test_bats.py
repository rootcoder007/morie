"""Tests for bats (De Livera, Hyndman & Snyder 2010: BATS and TBATS)."""

import importlib
import math

import pytest

from morie.fn._rng import random_normal

# morie.fn re-exports the bats() function under this name, so reach for
# the module rather than the function that shadows it.
M = importlib.import_module("morie.fn.bats")


# --------------------------------------------------------------------------
# equation 3a
# --------------------------------------------------------------------------

def test_box_cox_is_the_printed_transform():
    y = [3.0, 7.5, 1.25, 10.0]
    for i, v in enumerate(y):
        assert M.box_cox(y, 1.0)[i] == pytest.approx(v - 1.0, abs=1e-14)
        assert M.box_cox(y, 0.0)[i] == pytest.approx(math.log(v), abs=1e-14)
        assert M.box_cox(y, 0.5)[i] == pytest.approx(
            (v ** 0.5 - 1.0) / 0.5, abs=1e-14)


@pytest.mark.parametrize("w", [0.0, 0.3, 1.0, 1.4])
def test_box_cox_round_trips(w):
    y = [3.0, 7.5, 1.25, 10.0]
    back = M.inv_box_cox(M.box_cox(y, w), w)
    for i, v in enumerate(y):
        assert back[i] == pytest.approx(v, abs=1e-12)


def test_box_cox_needs_a_positive_series():
    with pytest.raises(ValueError):
        M.box_cox([1.0, -2.0], 0.5)
    with pytest.raises(ValueError):
        M.box_cox([1.0, 0.0], 0.5)
    with pytest.raises(ValueError):
        M.inv_box_cox([-100.0], 0.5)


# --------------------------------------------------------------------------
# the paper's own special case
# --------------------------------------------------------------------------

ALPHA, BETA, GAMMA = 0.3, 0.1, 0.2
LEV0, TR0 = 10.0, 0.5
SEAS0 = [1.5, -0.5, -2.0, 1.0]
Z = [12.0, 9.0, 8.5, 13.0, 12.6, 9.9, 8.2, 13.9, 13.1, 10.4, 9.0, 14.5]
SPEC = M.BatsSpec([4], None, use_trend=True, damped=False)
THETA = [ALPHA, BETA, GAMMA]
X0 = [LEV0, TR0] + SEAS0


def _holt_winters():
    lev, tr = LEV0, TR0
    seas = list(SEAS0)
    res, fit = [], []
    for zt in Z:
        pred = lev + tr + seas[0]
        e = zt - pred
        fit.append(pred)
        res.append(e)
        nl = lev + tr + ALPHA * e
        nt = tr + BETA * e
        old = seas.pop(0)
        seas.append(old + GAMMA * e)
        lev, tr = nl, nt
    return res, fit, lev, tr


def test_bats_1_1_0_0_m_is_holt_winters_additive():
    """The paper names this equivalence; it must hold exactly."""
    hw_res, hw_fit, hw_lev, hw_tr = _holt_winters()
    res, fit, carry = M.bats_filter(Z, SPEC, THETA, X0)
    for i in range(len(Z)):
        assert res[i] == pytest.approx(hw_res[i], abs=1e-13)
        assert fit[i] == pytest.approx(hw_fit[i], abs=1e-13)
    assert carry["level"] == pytest.approx(hw_lev, abs=1e-13)
    assert carry["trend"] == pytest.approx(hw_tr, abs=1e-13)
    assert SPEC.label() == "BATS(1, 1, 0, 0, 4)"


def test_the_filter_recovers_the_generating_innovations():
    eps = [float(v) for v in random_normal(60, seed=31, stream=0)]
    sp = M.BatsSpec([4], None, use_trend=True, damped=True, p=1, q=1)
    al, be, ph, g1, ar1, ma1 = 0.25, 0.08, 0.92, 0.15, 0.4, -0.3
    th = [al, be, ph, g1, ar1, ma1]
    x0 = [8.0, 0.3, 1.0, -0.4, -1.2, 0.6, 0.0, 0.0]
    lev, tr = x0[0], x0[1]
    buf = list(x0[2:6])
    dl, el = [x0[6]], [x0[7]]
    gen = []
    for e in eps:
        d = ar1 * dl[0] + ma1 * el[0] + e
        gen.append(lev + ph * tr + buf[0] + d)
        nl = lev + ph * tr + al * d
        nt = ph * tr + be * d
        old = buf.pop(0)
        buf.append(old + g1 * d)
        lev, tr = nl, nt
        dl, el = [d], [e]
    got, _, _ = M.bats_filter(gen, sp, th, x0)
    for i in range(len(eps)):
        assert got[i] == pytest.approx(eps[i], abs=1e-10)


# --------------------------------------------------------------------------
# the trigonometric seasonal, equations 4a-4c
# --------------------------------------------------------------------------

@pytest.mark.parametrize("m,k", [(4, 2), (7, 3), (12, 6)])
def test_the_index_equivalent_harmonic_count(m, k):
    """m/2 for even m, (m-1)/2 for odd -- the paper's equivalence."""
    assert len(M.seasonal_harmonics(m)) == k
    assert M.seasonal_harmonics(m)[0] == pytest.approx(2 * math.pi / m)


@pytest.mark.parametrize("m", [4, 7, 12])
def test_deterministic_seasonality_is_exactly_periodic(m):
    lam = M.seasonal_harmonics(m)
    sp = M.BatsSpec([m], [len(lam)], use_trend=False)
    nh = len(lam)
    seed = [0.0] + [1.0] * nh + [0.3] * nh
    _, fitted, _ = M.bats_filter([0.0] * (3 * m), sp, [0.0, 0.0, 0.0], seed)
    for t in range(len(fitted) - m):
        assert fitted[t] == pytest.approx(fitted[t + m], abs=1e-11)


def test_zero_gamma_makes_the_seasonal_recursion_a_rotation():
    m = 12
    lam = M.seasonal_harmonics(m)
    nh = len(lam)
    sp = M.BatsSpec([m], [nh], use_trend=False)
    seed = [0.0] + [0.7, -0.2, 1.1, 0.4, -0.9, 0.25] \
        + [0.1, 0.6, -0.3, 0.8, 0.05, -0.45]
    _, _, carry = M.bats_filter([0.0] * m, sp, [0.0, 0.0, 0.0], seed)
    for j in range(nh):
        assert carry["s"][0][j] == pytest.approx(seed[1 + j], abs=1e-11)
        assert carry["sstar"][0][j] == pytest.approx(seed[1 + nh + j],
                                                     abs=1e-11)
    _, _, half = M.bats_filter([0.0] * (m // 2), sp, [0.0, 0.0, 0.0], seed)
    assert half["s"][0][0] == pytest.approx(-seed[1], abs=1e-11)


def test_only_tbats_takes_a_fractional_period():
    with pytest.raises(ValueError, match="integer"):
        M.BatsSpec([365.25 / 7.0], None)
    sp = M.BatsSpec([365.25 / 7.0], [3])
    assert sp.trigonometric and sp.harmonics == [3]
    lam = M.seasonal_harmonics(365.25 / 7.0, 3)
    assert lam[0] == pytest.approx(2 * math.pi / (365.25 / 7.0), abs=1e-14)


def test_harmonics_above_half_the_period_are_refused():
    with pytest.raises(ValueError, match="alias"):
        M.seasonal_harmonics(12, 7)
    with pytest.raises(ValueError):
        M.seasonal_harmonics(12, 0)
    with pytest.raises(ValueError):
        M.seasonal_harmonics(1.0)


# --------------------------------------------------------------------------
# the seed state and the likelihood
# --------------------------------------------------------------------------

def test_the_seed_state_minimises_the_sum_of_squares():
    seed = M.fit_seed_state(Z, SPEC, THETA)
    r0, _, _ = M.bats_filter(Z, SPEC, THETA, seed)
    sse0 = sum(v * v for v in r0)
    for j in range(len(seed)):
        for delta in (-0.05, 0.05):
            pert = list(seed)
            pert[j] += delta
            rp, _, _ = M.bats_filter(Z, SPEC, THETA, pert)
            assert sum(v * v for v in rp) >= sse0 - 1e-12
    r_true, _, _ = M.bats_filter(Z, SPEC, THETA, X0)
    assert sse0 <= sum(v * v for v in r_true) + 1e-12


def test_the_likelihood_carries_the_box_cox_jacobian():
    y = [10.0, 12.0, 9.0, 11.0]
    r = [0.1, -0.2, 0.05, 0.3]
    base = M.concentrated_loglik(y, r, 1.0)
    assert base == pytest.approx(-0.5 * 4 * math.log(sum(v * v for v in r)))
    got = M.concentrated_loglik(y, r, 0.5)
    assert got - base == pytest.approx(-0.5 * sum(math.log(v) for v in y),
                                       abs=1e-12)
    worse = M.concentrated_loglik(y, [2 * v for v in r], 1.0)
    assert worse < base


# --------------------------------------------------------------------------
# the forecastability region
# --------------------------------------------------------------------------

def test_the_structural_unit_root_is_excluded_but_the_rest_is_not():
    sp = M.BatsSpec([12], None, use_trend=True, damped=False)
    # exactly one eigenvalue sits at 1 whatever the parameters: the
    # level and the seasonal indices are identified only up to a shift
    for th in ([0.2, 0.02, 0.1], [0.1, 0.05, 0.005], [0.3, 0.05, -0.5]):
        ev = M.all_eigenvalues(sp, th)
        assert sum(1 for v in ev if abs(v - 1.0) < 1e-6) == 1
    # and with it excluded the region still discriminates
    assert M.is_forecastable(sp, [0.2, 0.02, 0.1])
    assert not M.is_forecastable(sp, [0.3, 0.05, -0.5])
    assert M.spectral_radius(sp, [0.3, 0.05, -0.5]) > 1.0


def test_the_measurement_and_transition_matrices_reproduce_the_filter():
    w, fmat, g = M.state_matrices(SPEC, THETA)
    # w' x_0 must be the first one-step prediction
    _, fit, _ = M.bats_filter(Z, SPEC, THETA, X0)
    assert sum(w[j] * X0[j] for j in range(len(w))) == pytest.approx(
        fit[0], abs=1e-12)
    assert len(fmat) == len(g) == SPEC.n_states()


# --------------------------------------------------------------------------
# fitting and forecasting
# --------------------------------------------------------------------------

def _generate(gamma_true, n=200, sigma=1.5, seed=13):
    eps = [sigma * float(v) for v in random_normal(n, seed=seed, stream=3)]
    lev, tr = 100.0, 0.4
    sb = [8.0 * math.sin(2 * math.pi * j / 12.0) for j in range(12)]
    out = []
    for e in eps:
        out.append(lev + tr + sb[0] + e)
        nl = lev + tr + 0.20 * e
        nt = tr + 0.02 * e
        old = sb.pop(0)
        sb.append(old + gamma_true * e)
        lev, tr = nl, nt
    return out


def test_a_fit_recovers_sigma_and_alpha_and_stays_forecastable():
    # n = 400, not 200: fourteen seed values are concentrated out of the
    # likelihood, so on a short series the residual sd is biased down by
    # roughly sqrt(1 - k/n) and 200 points is not a fair sample to ask
    # recovery of.
    fit = M.bats(_generate(0.1, n=400), [12], use_box_cox=False,
                 use_trend=True,
                 damped=False, h=24)
    assert math.sqrt(fit["sigma2"]) == pytest.approx(1.5, abs=0.2)
    assert fit["alpha"] == pytest.approx(0.20, abs=0.1)
    assert fit["forecastable"]
    assert fit["spectral_radius"] < 1.0
    assert len(fit["forecast"]) == 24


def test_gamma_responds_to_the_generating_gamma():
    """Not a recovery check.

    With the seed concentrated out, the profile likelihood on a weakly
    drifting season is monotone towards gamma = 0 even at the true
    alpha and beta, so asserting recovery there would assert something
    false. What must hold is that the estimate moves with the truth.
    """
    got = [M.bats(_generate(g), [12], use_box_cox=False, use_trend=True,
                  damped=False)["gamma"][0] for g in (0.0, 0.5)]
    assert got[0] < got[1]
    assert got[1] - got[0] > 0.05


def test_a_frozen_state_forecasts_exactly_periodically():
    sp = M.BatsSpec([4], None, use_trend=False)
    fc = M._forecast(sp, [0.0, 0.0], [5.0, 1.0, -2.0, 0.5, 1.5],
                     [0.0] * 8, 8)
    for i in range(4):
        assert fc[i] == pytest.approx(fc[i + 4], abs=1e-12)
    assert fc[0] == pytest.approx(6.0, abs=1e-12)


def test_forecast_validation_and_series_checks():
    with pytest.raises(ValueError, match="horizon"):
        M._forecast(SPEC, THETA, X0, Z, -1)
    assert M._forecast(SPEC, THETA, X0, Z, 0) == []
    with pytest.raises(ValueError, match="too short"):
        M.bats([1.0, 2.0], [])
    with pytest.raises(ValueError, match="two full"):
        M.bats([1.0] * 10, [12])
    with pytest.raises(ValueError):
        M.BatsSpec([4, 7], [2])
    with pytest.raises(ValueError):
        M.BatsSpec([4], None, p=-1)
