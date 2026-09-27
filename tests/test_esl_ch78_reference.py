"""ESL chapters 7-8 against R: AIC/BIC arithmetic, smooth.spline GCV, predict.lm, the Bayesian posterior formula."""

import math

from morie.fn import (
    bic_posterior_probs,
    esl_aic_score,
    esl_bic_score,
    esl_cross_entropy,
    esl_cv_score,
    esl_mdl,
    esl_one_se_rule,
)
from morie.fn.eslbbp import esl_bayes_basis_posterior
from morie.fn.eslbse import esl_basis_fit_se
from morie.fn.esldrp import esl_dirichlet_posterior
from morie.fn.eslgcv import esl_gcv
from morie.fn.eslpbt import esl_parametric_bootstrap
from morie.fn.eslpee import esl_linear_prediction_error
from morie.fn.eslvcb import esl_vc_bound


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_criteria():
    ll = -16.170902885115233  # logLik of the glm in test_esl_ch4_reference
    assert close(esl_aic_score(ll, 3)["estimate"], 38.341805770230465)
    assert close(esl_bic_score(ll, 3, 40)["estimate"], 43.40844413257227)
    assert close(esl_mdl(ll, [0.1, 0.2, 0.3], n=40)["mdl"], 43.40844413257227 / 2)
    b = bic_posterior_probs([40.1, 41.3, 45.0])
    e = [1.0, math.exp(-0.6), math.exp(-2.45)]
    assert all(close(u, v / sum(e)) for u, v in zip(b["probs"], e)) and close(b["value"], b["probs"][0])
    ce = esl_cross_entropy([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    assert close(ce["estimate"], -(math.log(0.7) + math.log(0.8) + math.log(0.5)) / 3)
    assert esl_one_se_rule([0.30, 0.25, 0.22, 0.23, 0.26], [0.02, 0.02, 0.03, 0.02, 0.02])["estimate"] == 1
    v = esl_vc_bound(0.1, 5, 100)
    assert close(v["epsilon"], 1.1130569562097425) and close(v["bound"], 1.3053964481594105)
    r = esl_vc_bound(0.2, 5, 100, task="regression")
    assert close(r["bound"], 0.39491778481159401) and close(r["practical_bound"], 0.37880951126675527)
    assert esl_vc_bound(0.2, 90, 100, task="regression")["bound"] == math.inf  # 1 - sqrt(eps) <= 0


def test_cv_loo_and_gcv():
    i = range(1, 31)
    X = [[math.sin(t), math.log(t)] for t in i]
    y = [1 + 2 * r[0] - 0.5 * r[1] + 0.3 * math.cos(5 * t) for r, t in zip(X, i)]
    assert close(esl_cv_score(X, y, k=30)["cv"], 0.052279983486360551, 1e-11)  # mean((e / (1 - h))^2) of lm
    rss, df = 2.134556931459368645, 9.319916349372281417  # smooth.spline(spar = 0.6)
    g = esl_gcv([math.sqrt(rss)] + [0.0] * 49, [0.0] * 50, df)
    assert close(g["gcv"], 0.064493221698418493)


def test_prediction_error_basis_bayes_dirichlet():
    X = [[1.0, math.sin(t), math.log(t)] for t in range(1, 21)]
    pe = esl_linear_prediction_error(X, [1, 0.3, 2], 0.5)
    assert close(pe["h_norm2"], 0.056426555710195003) and close(pe["err_x0"], 0.528213277855097529)
    avg = sum(esl_linear_prediction_error(X, r, 0.5)["variance"] for r in X) / 20
    assert close(avg, 3 * 0.5 / 20)  # eq 7.12: (p / N) sigma^2
    assert close(pe["in_sample_variance"], avg)
    x = sorted(((7 * t) % 47) / 47 * 3 for t in range(1, 51))
    y = [math.sin(2 * v) + 0.3 * math.cos(9 * t) for t, v in enumerate(x, 1)]

    def tp(v):
        return [1.0, v, v * v, max(v - 1, 0) ** 3, max(v - 2, 0) ** 3]

    H = [tp(v) for v in x]
    Hn = [tp(0.5), tp(2.2)]
    s = esl_basis_fit_se(H, y, Hn, divisor="N-p")  # predict(lm(y ~ H - 1), se.fit = TRUE)
    for a, b in zip(
        s["fit"] + s["se"], (0.868055909560151884, -0.925522096722922516, 0.053027322056522511, 0.060592266691323650)
    ):
        assert close(a, b)
    s0 = esl_basis_fit_se(H, y, Hn)
    assert close(s0["se"][0], s["se"][0] * math.sqrt(45 / 50))
    bp = esl_bayes_basis_posterior(H, y, 0.09, 2.0, Hnew=Hn)
    for a, b in zip(
        bp["mean"][:3] + bp["mu_sd"],
        (
            -0.012817755105172329,
            2.558936604922364744,
            -1.691688010698427114,
            0.070516476962409258,
            0.081728670405200821,
        ),
    ):
        assert close(a, b, 1e-11)
    bt = esl_parametric_bootstrap(H, y, Hn, B=4000, seed=3)
    assert max(abs(u / v - 1) for u, v in zip(bt["boot_se"], s0["se"])) < 0.06
    d = esl_dirichlet_posterior([12, 5, 3])
    assert all(close(m, c / 20) for m, c in zip(d["mean"], (12, 5, 3)))
    assert all(close(v, (c / 20) * (1 - c / 20) / 21) for v, c in zip(d["var"], (12, 5, 3)))
