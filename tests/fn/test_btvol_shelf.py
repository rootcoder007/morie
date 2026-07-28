# morie.fn -- test file (rootcoder007/morie)
"""The bootstrap/jackknife and range-volatility shelves.

The oracles are closed forms wherever one exists: the jackknife
variance of the mean equals s^2/n EXACTLY; the MLE variance's bias is
exactly -sigma^2/n and the bootstrap and jackknife must both find it
with the right SIGN; the out-of-bag fraction is the 0.632 complement;
Parkinson's constant makes the estimator unbiased for driftless GBM
and its ~4.9 efficiency over close-to-close is MEASURED; and the
noise variance of contaminated high-frequency returns is recovered to
a few percent while naive realized variance is two orders off.
"""

import numpy as np
import pytest

from morie.fn.bt632 import boot_632_estimator
from morie.fn.btbias import boot_bias_estimator
from morie.fn.btciratio import boot_ci_ratio
from morie.fn.btiid import boot_iid_resample
from morie.fn.btjkn import boot_jackknife
from morie.fn.btoob import boot_oob_error
from morie.fn.btvb import boot_var_estimator
from morie.fn.volgkr import vol_garman_klass
from morie.fn.volharm import vol_harmonic
from morie.fn.volnois import vol_noise_variance
from morie.fn.volpark import vol_parkinson


# ------------------------------------------------- bootstrap


def test_iid_bootstrap_se_of_the_mean_matches_theory():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3, scale=2, size=200)
    o = boot_iid_resample(x, np.mean, B=800)
    assert o["se"] == pytest.approx(2 / np.sqrt(200), rel=0.15)
    assert o["estimate"] == pytest.approx(float(x.mean()), rel=1e-12)
    lo, hi = o["ci_percentile"]
    assert lo < 3 < hi
    assert "EMPIRICAL" in o["consistency_caveat"]


def test_variance_from_replicates_agrees_with_the_resampler():
    rng = np.random.default_rng(1)
    x = rng.normal(size=150)
    b = boot_iid_resample(x, np.mean, B=400)
    v = boot_var_estimator(b["replicates"])
    assert v["se"] == pytest.approx(b["se"], rel=1e-12)
    assert v["denominator"] == "B - 1"
    with pytest.raises(ValueError, match="at least 2"):
        boot_var_estimator([1.0])
    with pytest.raises(ValueError, match="finite"):
        boot_var_estimator([1.0, np.nan])


def test_bootstrap_bias_finds_the_mle_variance_bias_with_the_right_sign():
    """The MLE variance is biased by exactly -sigma^2/n. The
    correction must move the estimate UP, away from the replicate
    mean -- the direction is the classic mistake."""
    rng = np.random.default_rng(2)
    x = rng.normal(scale=2, size=200)
    mlvar = lambda d: float(np.var(d))
    b = boot_iid_resample(x, mlvar, B=8000, seed=1)
    o = boot_bias_estimator(b["estimate"], b["replicates"])
    # the bias estimate's own Monte-Carlo sd is sqrt(2 s^4 / n / B)
    # ~ 0.005 here, so the tolerance is three of those, not a
    # percentage that pretends the estimate is deterministic
    mc_sd = np.sqrt(2 * np.var(x, ddof=1) ** 2 / 200 / 8000)
    assert o["bias"] == pytest.approx(-np.var(x, ddof=1) / 200,
                                      abs=3 * mc_sd + 2 / 200 ** 1.5)
    assert o["bias"] < 0
    assert o["corrected"] > o["estimate"]          # moves UP
    assert o["corrected"] == pytest.approx(
        2 * o["estimate"] - o["mean_replicate"], rel=1e-12)


def test_jackknife_variance_of_the_mean_is_exactly_s2_over_n():
    """Not asymptotic -- an algebraic identity, so the tolerance is
    machine precision."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    j = boot_jackknife(x, np.mean)
    assert j["variance"] == pytest.approx(np.var(x, ddof=1) / 80, rel=1e-12)
    assert j["estimate"] == pytest.approx(float(x.mean()), rel=1e-12)
    # pseudovalues of the mean ARE the observations
    assert j["pseudovalues"] == pytest.approx(x, rel=1e-10)


def test_jackknife_bias_of_the_mle_variance_is_exact():
    """(n-1)(mean(loo) - full) for the MLE variance equals
    -s^2/n exactly -- the jackknife removes O(1/n) bias completely
    for quadratic statistics."""
    rng = np.random.default_rng(5)
    x = rng.normal(scale=1.5, size=120)
    j = boot_jackknife(x, lambda d: float(np.var(d)))
    assert j["bias"] == pytest.approx(-np.var(x, ddof=1) / 120, rel=1e-9)
    assert j["corrected"] == pytest.approx(np.var(x, ddof=1), rel=1e-9)
    assert "median" in j["smoothness_caveat"]
    with pytest.raises(ValueError, match="at least 3"):
        boot_jackknife(np.array([1.0, 2.0]), np.mean)


def test_632_alias_reproduces_the_books_worked_numbers():
    o = boot_632_estimator(0.0, 0.5, gamma=0.5)
    assert o["err_632"] == pytest.approx(0.316, abs=1e-12)
    assert o["err_632_plus"] == pytest.approx(0.5, rel=1e-12)
    assert o["alias_of"] == "morie.fn.eslo63.esl_oob_632"


def test_oob_error_is_honest_and_the_oob_fraction_is_0368():
    rng = np.random.default_rng(7)
    n = 120
    X = rng.normal(size=(n, 2))
    y = X @ [1.0, -1.0] + rng.normal(scale=0.5, size=n)

    def fit(Xa, ya):
        return np.linalg.lstsq(np.column_stack([np.ones(len(ya)), Xa]),
                               ya, rcond=None)[0]

    def pred(b, Xn):
        return np.column_stack([np.ones(len(Xn)), Xn]) @ b

    o = boot_oob_error(X, y, fit, pred, B=100)
    # honesty: out-of-bag error exceeds the apparent (resubstitution)
    assert o["err_oob"] > o["err_apparent"]
    # each point is out of bag for about 1 - 0.632 of replicates
    assert o["oob_fraction"] == pytest.approx(1 - 0.632, abs=0.03)
    assert o["n_dropped"] == 0


def test_ratio_ci_covers_and_pairing_matters():
    rng = np.random.default_rng(9)
    a = 2.0 + rng.normal(scale=0.3, size=300)
    b = 1.0 + rng.normal(scale=0.2, size=300)
    o = boot_ci_ratio(a, b, B=1000)
    lo, hi = o["ci"]
    assert lo < 2.0 < hi
    assert o["ratio"] == pytest.approx(float(a.mean() / b.mean()), rel=1e-12)
    # strongly positively correlated numerator and denominator: the
    # ratio's variance shrinks, and PAIRED resampling must see that
    common = rng.normal(size=400)
    x2 = 2.0 + common + 0.05 * rng.normal(size=400)
    y2 = 1.0 + 0.5 * common + 0.05 * rng.normal(size=400)
    paired = boot_ci_ratio(x2, y2, B=800, paired=True, seed=3)
    indep = boot_ci_ratio(x2, y2, B=800, paired=False, seed=3)
    assert paired["se"] < indep["se"]
    with pytest.raises(ValueError, match="equal-length"):
        boot_ci_ratio(x2[:10], y2, paired=True)
    with pytest.raises(ValueError, match="at least 100"):
        boot_ci_ratio(a, b, B=10)


# ------------------------------------------------- volatility


def gbm_bars(n_bars, sigma, steps=390, seed=0):
    rng = np.random.default_rng(seed)
    O, H, L, C = [], [], [], []
    p = 0.0
    for _ in range(n_bars):
        path = p + np.cumsum(rng.normal(scale=sigma / np.sqrt(steps),
                                        size=steps))
        O.append(p)
        C.append(path[-1])
        H.append(max(p, path.max()))
        L.append(min(p, path.min()))
        p = path[-1]
    return (np.exp(np.array(O)), np.exp(np.array(H)),
            np.exp(np.array(L)), np.exp(np.array(C)))


def test_parkinson_is_unbiased_for_driftless_gbm():
    O, H, L, C = gbm_bars(400, 0.02, seed=0)
    o = vol_parkinson(H, L)
    assert o["sigma"] == pytest.approx(0.02, rel=0.08)
    assert o["constant"] == pytest.approx(1 / (4 * np.log(2)), rel=1e-12)
    with pytest.raises(ValueError, match="high must be at least low"):
        vol_parkinson([1.0, 1.0], [1.1, 0.9])
    with pytest.raises(ValueError, match="positive"):
        vol_parkinson([1.0, 1.0], [-0.5, 0.9])


def test_parkinson_efficiency_over_close_to_close_is_measured():
    """The ~4.9 variance reduction is the estimator's reason to
    exist, so it is measured over replications rather than quoted."""
    pk, cc = [], []
    for rep in range(60):
        O, H, L, C = gbm_bars(100, 0.02, seed=100 + rep)
        pk.append(vol_parkinson(H, L)["variance"])
        r = np.diff(np.log(C), prepend=0.0)
        cc.append(float(np.var(r, ddof=1)))
    ratio = np.var(cc) / np.var(pk)
    assert 3.0 < ratio < 8.0


def test_garman_klass_beats_parkinson_and_partials_out_trend():
    O, H, L, C = gbm_bars(400, 0.02, seed=1)
    g = vol_garman_klass(O, H, L, C)
    assert g["sigma"] == pytest.approx(0.02, rel=0.08)
    # the open-close term is genuinely SUBTRACTED
    assert g["variance"] == pytest.approx(
        g["range_term"] - g["openclose_term"], rel=1e-10)
    # measured efficiency: GK spread below Parkinson's
    gk, pk = [], []
    for rep in range(60):
        Ob, Hb, Lb, Cb = gbm_bars(100, 0.02, seed=300 + rep)
        gk.append(vol_garman_klass(Ob, Hb, Lb, Cb)["variance"])
        pk.append(vol_parkinson(Hb, Lb)["variance"])
    assert np.var(gk) < np.var(pk)
    with pytest.raises(ValueError, match="low <= open"):
        vol_garman_klass([2.0, 1.0], [1.5, 1.2], [0.9, 0.8], [1.0, 1.0])


def test_harmonic_mean_inequality_and_use_guidance():
    o = vol_harmonic([0.1, 0.2, 0.4])
    assert o["inequality_holds"] is True
    assert o["harmonic"] < o["geometric"] < o["arithmetic"] < o["rms"]
    assert o["harmonic"] == pytest.approx(3 / (10 + 5 + 2.5), rel=1e-12)
    # equal inputs collapse all four
    e = vol_harmonic([0.3, 0.3, 0.3])
    assert e["harmonic"] == pytest.approx(e["rms"], rel=1e-12)
    assert "ARITHMETIC" in o["which_to_use"].upper() or \
        "arithmetic" in o["which_to_use"]
    with pytest.raises(ValueError, match="positive"):
        vol_harmonic([0.1, 0.0])


def test_noise_variance_is_recovered_and_naive_rv_diverges():
    """The ZMA decomposition E[RV] = IV + 2n eps^2: the noise term is
    recovered to a few percent, the two-scale IV is near the truth,
    and the naive RV is two orders of magnitude off -- the signature
    plot in one test."""
    rng = np.random.default_rng(11)
    n = 23_400
    iv = 0.01 ** 2
    dP = rng.normal(scale=np.sqrt(iv / n), size=n)
    eps = rng.normal(scale=5e-4, size=n + 1)
    r = dP + np.diff(eps)
    o = vol_noise_variance(r)
    assert o["noise_variance"] == pytest.approx((5e-4) ** 2, rel=0.1)
    assert o["iv_two_scale"] == pytest.approx(iv, rel=0.5)
    assert o["rv_all"] > 50 * iv                     # the divergence
    assert o["noise_share_of_rv"] > 0.9
    # clean returns: tiny noise estimate, TSRV near naive RV
    clean = vol_noise_variance(dP)
    assert clean["noise_variance"] < o["noise_variance"] / 20
    with pytest.raises(ValueError, match="at least 30"):
        vol_noise_variance(r[:10])
    with pytest.raises(ValueError, match="K must lie"):
        vol_noise_variance(r, K=1)
