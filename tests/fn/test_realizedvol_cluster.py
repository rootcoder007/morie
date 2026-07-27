"""Realized/range volatility cluster: volrm volsd volrv volbpv volrk
volrs volyz volhar volhar1 volharj volraq voldoc voltsr voljr volpow
volmuk volopn volsk volrls volrlmt."""

import numpy as np
import pytest

from morie.fn.volbpv import vol_bipower_variation
from morie.fn.voldoc import vol_decomposed_realised
from morie.fn.volhar import vol_har_rv
from morie.fn.volhar1 import vol_har_q
from morie.fn.volharj import vol_har_rv_jump
from morie.fn.voljr import vol_jump_robust_var
from morie.fn.volmuk import vol_multi_kernel_rk
from morie.fn.volopn import vol_implied_volatility_bs, _bs_price
from morie.fn.volpow import vol_power_variation
from morie.fn.volraq import vol_realised_quadratic_var
from morie.fn.volrk import vol_realised_kernel
from morie.fn.volrlmt import vol_realised_log_vol_ar
from morie.fn.volrls import vol_recursive_least_sq
from morie.fn.volrm import vol_riskmetrics
from morie.fn.volrs import vol_rogers_satchell
from morie.fn.volrv import vol_realised_variance
from morie.fn.volsd import vol_simple_diff
from morie.fn.volsk import vol_stochastic_kalman
from morie.fn.voltsr import vol_two_scale_rv
from morie.fn.volyz import vol_yang_zhang


def test_volrm_hand_recursion():
    r = np.array([1.0, 2.0, -1.0, 0.5] + [1.0] * 20)
    out = vol_riskmetrics(r, lam=0.9)
    s0 = out["sigma2"][0]
    assert out["sigma2"][1] == pytest.approx(0.9 * s0 + 0.1 * 1.0)
    assert out["sigma2"][2] == pytest.approx(0.9 * out["sigma2"][1] + 0.1 * 4.0)
    assert out["lam"] == 0.9
    assert vol_riskmetrics(r)["lam"] == 0.94  # the RM 1996 p.51 default
    with pytest.raises(ValueError):
        vol_riskmetrics(r, lam=1.0)


def test_volsd_rolling():
    r = np.array([1.0, 1.0, 1.0, 3.0])
    out = vol_simple_diff(r, window=2)
    assert np.isnan(out["sigma"][0])
    assert out["sigma2"][1] == pytest.approx(1.0)
    assert out["sigma2"][3] == pytest.approx((1.0 + 9.0) / 2)
    with pytest.raises(ValueError):
        vol_simple_diff(r, window=10)


def test_volrv_and_per_day():
    r = np.array([0.1, -0.2, 0.3, 0.1])
    assert vol_realised_variance(r)["rv"] == pytest.approx(0.15)
    days = ["a", "a", "b", "b"]
    out = vol_realised_variance(r, days)
    assert out["rv"] == pytest.approx([0.05, 0.10])
    assert out["days"] == ["a", "b"]


def test_volbpv_jump_robustness():
    rng = np.random.default_rng(0)
    m = 400
    sig = 0.01
    r = rng.normal(scale=sig, size=m)
    true_iv = sig**2 * m
    clean_rv = vol_realised_variance(r)["rv"]
    clean_bpv = vol_bipower_variation(r)["bpv"]
    assert clean_bpv == pytest.approx(true_iv, rel=0.2)
    rj = r.copy()
    rj[200] += 0.2  # one large jump
    rv_j = vol_realised_variance(rj)["rv"]
    bpv_j = vol_bipower_variation(rj)["bpv"]
    # the jump inflates RV by ~0.04 but BPV only marginally
    assert rv_j - clean_rv > 0.03
    assert bpv_j - clean_bpv < 0.01
    with pytest.raises(ValueError):
        vol_bipower_variation([0.1, 0.2])


def test_volraq_and_voldoc():
    x = np.array([0.0, 1.0, 3.0, 2.0])
    out = vol_realised_quadratic_var(x)
    assert out["qv"] == pytest.approx(1 + 4 + 1)
    assert out["rq"] == pytest.approx(3 / 3 * (1 + 16 + 1))
    d = vol_decomposed_realised([2.0, 1.0], [1.5, 1.2])
    assert d["jump"] == pytest.approx([0.5, 0.0])  # negative diff truncated
    assert d["continuous"] == pytest.approx([1.5, 1.0])
    with pytest.raises(ValueError):
        vol_decomposed_realised([1.0], [1.0, 2.0])


def _noisy_returns(seed, m=1000, sig=0.01, noise_sd=0.004):
    rng = np.random.default_rng(seed)
    p_true = np.cumsum(rng.normal(scale=sig, size=m))
    noise = rng.normal(scale=noise_sd, size=m)
    return np.diff(np.concatenate([[0.0], p_true + noise])), sig**2 * m


def test_volrk_noise_correction():
    # measured over seeds 0-5: RV sits 30-40% above the truth, the
    # Bartlett kernel lands closer in 6/6 (rk errors 0.00-0.045 vs
    # rv errors 0.031-0.036); assert improvement as a rate, not a
    # single-seed factor
    hits = 0
    for seed in range(6):
        r, iv = _noisy_returns(seed)
        rv = vol_realised_variance(r)["rv"]
        rk = vol_realised_kernel(r)["rk"]
        assert rv > 1.2 * iv  # the noise bias the kernel must fix
        hits += abs(rk - iv) < abs(rv - iv)
    assert hits >= 5
    r, _ = _noisy_returns(0)
    with pytest.raises(ValueError):
        vol_realised_kernel(r, H=r.size)


def test_voltsr_noise_correction():
    hits = 0
    for seed in range(6):
        r, iv = _noisy_returns(seed)
        out = vol_two_scale_rv(r, K=15)
        hits += abs(out["tsrv"] - iv) < abs(out["rv_fast"] - iv)
    assert hits >= 5  # measured 6/6 at K = 15
    r, _ = _noisy_returns(0)
    with pytest.raises(ValueError):
        vol_two_scale_rv(r[:5], K=10)


def test_volmuk_matches_rk_on_clean_data():
    rng = np.random.default_rng(3)
    r = rng.normal(scale=0.01, size=600)
    avg = vol_multi_kernel_rk(r, n_grids=3)
    assert avg["rk_per_grid"].size == 3
    # clean data: subgrid kernels all near the true IV
    assert avg["rk_avg"] == pytest.approx(0.01**2 * 600, rel=0.3)
    with pytest.raises(ValueError):
        vol_multi_kernel_rk(r[:8], n_grids=3)


def test_volrs_hand_and_flat_bar():
    # flat bar: zero variance by construction
    flat = vol_rogers_satchell([10.0], [10.0], [10.0], [10.0])
    assert flat["sigma2"][0] == pytest.approx(0.0)
    out = vol_rogers_satchell([100.0], [110.0], [95.0], [105.0])
    expect = np.log(110 / 105) * np.log(110 / 100) + np.log(95 / 105) * np.log(95 / 100)
    assert out["sigma2"][0] == pytest.approx(expect)
    with pytest.raises(ValueError):
        vol_rogers_satchell([100.0], [99.0], [95.0], [105.0])  # h < c


def test_volyz_recovers_sigma():
    rng = np.random.default_rng(4)
    n, steps = 300, 50
    sig_step = 0.02 / np.sqrt(steps)
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    price = 0.0
    for d in range(n):
        o[d] = price
        path = price + np.cumsum(rng.normal(scale=sig_step, size=steps))
        h[d] = max(path.max(), price)
        l[d] = min(path.min(), price)
        c[d] = path[-1]
        price = c[d]
    O, H, L, C = (np.exp(v) for v in (o, h, l, c))
    out = vol_yang_zhang(O, H, L, C)
    assert out["sigma"] == pytest.approx(0.02, rel=0.2)  # daily sigma
    assert out["k"] == pytest.approx(0.34 / (1.34 + (n + 1) / (n - 1)))
    with pytest.raises(ValueError):
        vol_yang_zhang(O[:1], H[:1], L[:1], C[:1])


def _har_sim(seed, n=600):
    rng = np.random.default_rng(seed)
    rv = np.empty(n)
    rv[:22] = 1.0
    for t in range(22, n):
        m = 0.05 + 0.4 * rv[t - 1] + 0.3 * rv[t - 5 : t].mean() + 0.2 * rv[t - 22 : t].mean()
        rv[t] = max(m + rng.normal(scale=0.05), 1e-4)
    return rv


def test_volhar_recovers_cascade():
    hits = 0
    for seed in range(6):
        rv = _har_sim(seed)
        out = vol_har_rv(rv, h=3)
        c, bd, bw, bm = out["coefficients"]
        hits += abs(bd - 0.4) < 0.15 and abs(bw - 0.3) < 0.35 and out["r2"] > 0.3
        assert out["forecast"].size == 3
        assert np.all(out["forecast"] >= 0)
    assert hits >= 5  # measured 6/6
    with pytest.raises(ValueError):
        vol_har_rv(rv[:10])


def test_volhar1_and_volharj_nest_har():
    rv = _har_sim(1)
    rng = np.random.default_rng(1)
    rq = rv**2 * (1 + rng.random(rv.size))
    bpv = rv * (0.8 + 0.2 * rng.random(rv.size))
    base = vol_har_rv(rv)["r2"]
    q = vol_har_q(rv, rq)
    j = vol_har_rv_jump(rv, bpv)
    assert q["coefficients"].size == 5
    assert j["coefficients"].size == 5
    # nested models: in-sample R2 can only improve
    assert q["r2"] >= base - 1e-10
    assert j["r2"] >= base - 1e-10
    with pytest.raises(ValueError):
        vol_har_q(rv, rq[:10])


def test_voljr_excludes_the_jump():
    rng = np.random.default_rng(5)
    r = rng.normal(scale=0.01, size=300)
    r[100] = 0.3
    out = vol_jump_robust_var(r)
    assert out["n_excluded"] >= 1
    assert out["rv"] - out["rv_truncated"] > 0.08  # the jump's square
    with pytest.raises(ValueError):
        vol_jump_robust_var(r, threshold=-1.0)


def test_volpow_p2_is_rv():
    rng = np.random.default_rng(6)
    r = rng.normal(scale=0.01, size=200)
    out = vol_power_variation(r, p=2.0)
    # mu_2 = 1 and m^0 = 1, so the standardised PV at p = 2 IS the RV
    assert out["mu_p"] == pytest.approx(1.0)
    assert out["pv_standardised"] == pytest.approx((r**2).sum())
    with pytest.raises(ValueError):
        vol_power_variation(r, p=0.0)


def test_volopn_round_trip():
    S, K, T, r = 100.0, 105.0, 0.5, 0.02
    for sigma in (0.15, 0.45):
        price = _bs_price(S, K, T, r, sigma, "call")
        out = vol_implied_volatility_bs(S, K, T, r, price, "call")
        assert out["implied_vol"] == pytest.approx(sigma, abs=1e-8)
        assert out["vega"] > 0
    pput = _bs_price(S, K, T, r, 0.3, "put")
    assert vol_implied_volatility_bs(S, K, T, r, pput, "put")["implied_vol"] == pytest.approx(0.3, abs=1e-8)
    with pytest.raises(ValueError):
        vol_implied_volatility_bs(S, K, T, r, S + 1.0, "call")  # above the bound


def test_volsk_tracks_a_regime_shift():
    rng = np.random.default_rng(7)
    r = np.concatenate([
        rng.normal(scale=0.01, size=300),
        rng.normal(scale=0.04, size=300),
    ])
    out = vol_stochastic_kalman(r)
    assert out["sigma"][350:].mean() > 2.0 * out["sigma"][:250].mean()
    with pytest.raises(ValueError):
        vol_stochastic_kalman(r, phi=1.5)


def test_volrls_hand_recursion_and_convention():
    r = np.array([1.0, 2.0] + [1.0] * 20)
    out = vol_recursive_least_sq(r, lam=0.9)
    s0 = out["sigma2"][0]
    # nowcast convention: uses r_t, not r_{t-1}
    assert out["sigma2"][1] == pytest.approx(0.9 * s0 + 0.1 * 4.0)
    rm = vol_riskmetrics(r, lam=0.9)
    assert rm["sigma2"][1] == pytest.approx(0.9 * rm["sigma2"][0] + 0.1 * 1.0)
    assert out["effective_window"] == pytest.approx(10.0)


def test_volrlmt_recovers_phi():
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 500
        y = np.empty(n)
        y[0] = 0.0
        for t in range(1, n):
            y[t] = 0.1 + 0.7 * y[t - 1] + rng.normal(scale=0.3)
        out = vol_realised_log_vol_ar(np.exp(y), h=2)
        hits += abs(out["phi"] - 0.7) < 0.1
        assert out["forecast"].size == 2
        assert np.all(out["forecast"] > 0)
    assert hits >= 5  # measured 6/6
    with pytest.raises(ValueError):
        vol_realised_log_vol_ar(np.array([1.0, -1.0] * 10))
