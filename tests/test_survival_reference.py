"""morie.survival against R survival / survRM2 on one deterministic data set.

References (R): coxph(..., control = coxph.control(eps = 1e-15)) for
ties = "efron" and "breslow"; survdiff(rho = 1); survRM2:::rmst1(tau = 8);
concordance(Surv(t, e) ~ x1, reverse = TRUE).
"""

import math

from morie import survival as S
from morie.fn import _frame_core as pd


def _data():
    T, E, G, X1, X2 = [], [], [], [], []
    for i in range(150):
        x1 = math.sin(1.3 * i)
        x2 = 1.0 if math.cos(0.9 * i) > 0 else 0.0
        g = i % 2
        rate = math.exp(0.4 * x1 - 0.5 * x2 + 0.3 * g)
        t = round(-math.log(0.02 + 0.96 * abs(math.sin(2.3 * i + 0.4))) / rate * 5, 1)
        c = round(2 + 8 * abs(math.cos(1.9 * i)), 1)
        T.append(max(min(t, c), 0.1))
        E.append(1 if t <= c else 0)
        G.append(g)
        X1.append(x1)
        X2.append(x2)
    return T, E, G, X1, X2


def rel(a, b):
    return abs(a - b) / abs(b)


def test_cox_efron_and_breslow_match_coxph():
    T, E, _, X1, X2 = _data()
    df = pd.DataFrame({"t": T, "e": E, "x1": X1, "x2": X2})
    ef = S.cox_ph(df, "t", "e", ["x1", "x2"], ties="efron")
    for got, ref in zip(
        list(ef.coefficients) + list(ef.standard_errors),
        [0.39683246745802891, -0.28709499700636076, 0.13311968584722475, 0.18038003077936654],
    ):
        assert rel(float(got), ref) <= 1e-10
    assert rel(ef.log_likelihood, -522.54206557445002) <= 1e-13
    br = S.cox_ph(df, "t", "e", ["x1", "x2"], ties="breslow")
    for got, ref in zip(
        list(br.coefficients) + list(br.standard_errors),
        [0.38774134800461113, -0.27843807112528629, 0.13295767792607099, 0.18037205662333858],
    ):
        assert rel(float(got), ref) <= 1e-10


def test_peto_peto_is_survdiff_rho_1():
    T, E, G, _, _ = _data()
    assert rel(S.peto_peto_test(T, E, G).test_statistic, 1.064620461630706) <= 1e-12


def test_rmst_matches_survrm2():
    T, E, _, _, _ = _data()
    r = S.restricted_mean_survival_time(T, E, tau=8)
    assert rel(r["rmst"], 2.9384051107858156) <= 1e-13
    assert rel(r["se"], 0.2285745202098439) <= 1e-12


def test_concordance_matches_survival():
    T, E, _, X1, _ = _data()
    assert rel(S.concordance_index(T, X1, E), 0.56794871794871793) <= 1e-13


# survreg(Surv(t, e) ~ x1 + x2, dist = ..., control = survreg.control(rel.tolerance = 1e-14)):
# coefficients, scale, SEs of (intercept, x1, x2, log scale), log-likelihood
AFT_REF = {
    "weibull": (
        [0.95464566434250653, -0.42397359150975616, 0.3596205839009648],
        1.2082781490838239,
        [0.14915168219246222, 0.15600025833897435, 0.21785133436179238, 0.073715052052702901],
        -262.92038894138722,
    ),
    "lognormal": (
        [0.2930708670275064, -0.44501849394716181, 0.40973903203989143],
        1.5231352509549119,
        [0.176947723904718, 0.18112984231016344, 0.25486483101727631, 0.06506216323184047],
        -263.31995645449791,
    ),
    "loglogistic": (
        [0.35322548245078, -0.40628730573015726, 0.39980424072509529],
        0.90923066767639271,
        [0.18417134590609704, 0.18851426245724751, 0.26365034787765024, 0.073649829110218265],
        -266.62828968729383,
    ),
}


def test_aft_models_match_survreg():
    T, E, _, X1, X2 = _data()
    df = pd.DataFrame({"t": T, "e": E, "x1": X1, "x2": X2})
    for dist, fn in (("weibull", S.aft_weibull), ("lognormal", S.aft_lognormal), ("loglogistic", S.aft_loglogistic)):
        coef, scale, se, ll = AFT_REF[dist]
        r = fn(df, "t", "e", ["x1", "x2"])
        got = [r.coefficients["intercept"], r.coefficients["x1"], r.coefficients["x2"]]
        for a, b in zip(got, coef):
            assert rel(a, b) <= 1e-11
        assert rel(r.coefficients["sigma"], scale) <= 1e-11
        for a, b in zip(r.extra["se"], se):
            assert rel(a, b) <= 1e-10
        assert rel(r.log_likelihood, ll) <= 1e-13
