"""IPW/propensity cluster: causipsw, cipsc, causmtch, causqte, nonresp,
unitnr, spwgts, msmest, prsmtd.

Assertions are hand-computable identities plus parameter recovery under
known DGPs, with rates over seeds for anything stochastic."""

from morie.fn import _array_core as np
import pytest

from morie.fn.aiptdd import _logit_fit
from morie.fn.causipsw import causal_iptw_attweights
from morie.fn.causmtch import causal_pair_matching
from morie.fn.causqte import causal_quantile_treatment_effect
from morie.fn.cipsc import caliper_psm
from morie.fn.msmest import marginal_structural_model
from morie.fn.nonresp import nonresponse_adjustment
from morie.fn.prsmtd import propensity_score_method
from morie.fn.spwgts import spline_weights
from morie.fn.unitnr import unit_nonresponse


def test_att_weights_hand():
    out = causal_iptw_attweights([1, 0, 0], [0.5, 0.25, 0.5])
    # treated -> 1; controls -> e/(1-e) = 1/3 and 1
    assert out["weights"] == pytest.approx([1.0, 1.0 / 3.0, 1.0])
    # ESS of control weights (1/3, 1): (4/3)^2 / (1/9 + 1) = 1.6
    assert out["ess_control"] == pytest.approx(1.6)
    with pytest.raises(ValueError):
        causal_iptw_attweights([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        causal_iptw_attweights([0, 1], [1.0, 0.5])


def test_att_weights_balance_ps():
    # ATT weights make the weighted control ps distribution match the treated one.
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    e = 1 / (1 + np.exp(-x))
    T = (rng.random(4000) < e).astype(float)
    w = causal_iptw_attweights(T, e)["weights"]
    c = T == 0
    wmean_ctrl = np.sum(w[c] * e[c]) / np.sum(w[c])
    # measured gap ~0.004 at n=4000
    assert wmean_ctrl == pytest.approx(e[T == 1].mean(), abs=0.03)


def test_caliper_psm_hand():
    e = np.array([0.80, 0.79, 0.30, 0.60, 0.05])
    T = np.array([1, 0, 1, 0, 0])
    y = np.array([5.0, 1.0, 3.0, 2.0, 0.0])
    out = caliper_psm(e, T, caliper=0.5, y=y)  # logit-scale caliper
    pairs = {tuple(p) for p in out["matched_idx"]}
    # 0.80 matches 0.79 (logit gap 0.06); 0.30 has nearest remaining 0.60
    # (logit gap 1.25 > 0.5) and 0.05 (gap 2.10) -> unmatched
    assert pairs == {(0, 1)}
    assert out["att"] == pytest.approx(4.0)
    assert out["n_treated"] == 2
    # default caliper is 0.2*sd(logit e)
    d = caliper_psm(e, T)
    lg = np.log(e / (1 - e))
    assert d["caliper"] == pytest.approx(0.2 * lg.std(ddof=1))


def test_nn_matching_att_recovery():
    # y = 2*T + x; matching on true ps must remove the confounding by x.
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=1500)
        e = 1 / (1 + np.exp(-1.2 * x))
        T = (rng.random(1500) < e).astype(float)
        y = 2.0 * T + x + rng.normal(scale=0.5, size=1500)
        naive = y[T == 1].mean() - y[T == 0].mean()  # measured ~3.3: badly biased
        out = causal_pair_matching(e, T, caliper=0.1, y=y)
        hits += abs(out["att"] - 2.0) < 0.25
        assert abs(naive - 2.0) > 0.5
    assert hits >= 7  # measured 8/8


def test_qte_constant_shift():
    # Y(1) = Y(0) + 3 everywhere -> QTE(tau) = 3 at every tau.
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=3000)
        e = np.clip(1 / (1 + np.exp(-x)), 0.05, 0.95)
        T = (rng.random(3000) < e).astype(float)
        y0 = x + rng.normal(size=3000)
        y = y0 + 3.0 * T
        out = causal_quantile_treatment_effect(y, T, e, tau=[0.25, 0.5, 0.75])
        hits += np.all(np.abs(out["qte"] - 3.0) < 0.35)
    assert hits >= 7  # measured 8/8


def test_qte_weighted_quantile_reduces_to_ordinary():
    y = np.arange(1.0, 101.0)
    T = np.r_[np.ones(50), np.zeros(50)]
    ps = np.full(100, 0.5)
    out = causal_quantile_treatment_effect(y, T, ps, tau=0.5)
    assert out["q1"] == pytest.approx(np.sort(y[:50])[24], abs=1.0)
    assert out["q0"] == pytest.approx(np.sort(y[50:])[24], abs=1.0)


def test_nonresp_hand():
    out = nonresponse_adjustment([1.0, 2.0], [1.0, 1.0], [0.5, 1.0])
    # adjusted weights (2, 1); Hajek mean = (2*1 + 1*2)/3
    assert out["weights_adjusted"] == pytest.approx([2.0, 1.0])
    assert out["estimate"] == pytest.approx(4.0 / 3.0)
    assert out["ess"] == pytest.approx(9.0 / 5.0)
    with pytest.raises(ValueError):
        nonresponse_adjustment([1.0], [1.0], [1.5])


def test_unitnr_debiases_mcar_violation():
    # Response depends on x, y depends on x: respondent mean is biased,
    # propensity weighting recovers the frame mean.
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=3000)
        y = 2.0 + 1.5 * x + rng.normal(scale=0.5, size=3000)
        phi = 1 / (1 + np.exp(-(0.5 + 1.5 * x)))
        r = (rng.random(3000) < phi).astype(float)
        out = unit_nonresponse(r, None, x, y=y)
        naive = y[r == 1].mean()
        assert abs(naive - 2.0) > 0.3  # measured bias ~0.75
        hits += abs(out["estimate"] - 2.0) < 0.25
    assert hits >= 7  # measured 8/8
    assert out["weights"][r == 0].max() == 0.0


def test_spline_weights_nonlinear_propensity():
    # e(x) is U-shaped in x -- the confounder is x^2, which a linear
    # logit cannot represent but the spline basis can.
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=4000)
        e = np.clip(1 / (1 + np.exp(-(x**2 - 1.0))), 0.05, 0.95)
        A = (rng.random(4000) < e).astype(float)
        out = spline_weights(A, x)
        w = out["weights"]

        def gap(wts, v):  # weighted between-arm gap in v
            g1 = np.sum(wts[A == 1] * v[A == 1]) / np.sum(wts[A == 1])
            g0 = np.sum(wts[A == 0] * v[A == 0]) / np.sum(wts[A == 0])
            return abs(g1 - g0)

        x2 = x**2
        raw = abs(x2[A == 1].mean() - x2[A == 0].mean())
        assert raw > 0.5  # measured ~1.07 every seed
        # spline + Cole-Hernan truncation: measured gaps 0.08-0.24
        assert gap(w, x2) < 0.35
        assert out["ess"] < 4000.0
        # linear-logit weights leave the x^2 imbalance untouched
        # (measured gap == raw to 3 decimals every seed)
        lin = np.clip(_logit_fit(x[:, None], A), 0.01, 0.99)
        wl = A / lin + (1 - A) / (1 - lin)
        assert gap(w, x2) < gap(wl, x2)
        assert gap(wl, x2) > 0.5


def test_msm_point_treatment_recovery():
    # One period: MSM by IPTW must undo confounding by L.
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        L = rng.normal(size=2500)
        e = 1 / (1 + np.exp(-1.5 * L))
        A = (rng.random(2500) < e).astype(float)
        y = 2.0 * A + 1.5 * L + rng.normal(scale=0.5, size=2500)
        out = marginal_structural_model(y, A, L)
        naive = y[A == 1].mean() - y[A == 0].mean()
        assert abs(naive - 2.0) > 0.5  # measured ~3.7
        hits += abs(out["estimate"] - 2.0) < 0.3
    assert hits >= 7  # measured 8/8


def test_msm_two_periods_runs_and_weights_stabilised():
    rng = np.random.default_rng(0)
    n = 2000
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.5 * A1 + rng.normal(size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-L2))).astype(float)
    y = 1.0 * (A1 + A2) + L1 + rng.normal(size=n)
    out = marginal_structural_model(y, np.c_[A1, A2], np.c_[L1, L2])
    # stabilised weights average ~1 by construction (measured 1.00)
    assert out["weights"].mean() == pytest.approx(1.0, abs=0.15)
    assert out["estimate"] == pytest.approx(1.0, abs=0.35)  # measured ~1.05


def test_sequential_matching_structure():
    rng = np.random.default_rng(4)
    n, T = 400, 3
    H = rng.normal(size=(n, T))
    A = np.zeros((n, T))
    ever = np.zeros(n, dtype=bool)
    for t in range(T):
        p = 1 / (1 + np.exp(-H[:, t]))
        start = (rng.random(n) < 0.3 * p) & ~ever
        A[start, t] = 1
        ever |= start
    out = propensity_score_method(A, H)
    m = out["matched_idx"]
    assert m.shape[0] > 0
    for t, i, j in m:
        assert A[i, t] == 1
        assert A[j, : t + 1].sum() == 0  # control still untreated through t
        assert A[i, :t].sum() == 0  # initiator newly treated at t
    # no control reused within the run
    assert len({(j) for _, _, j in m}) == m.shape[0]
    assert out["n_matched"] <= out["n_initiators"]
