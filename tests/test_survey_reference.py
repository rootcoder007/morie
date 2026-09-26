"""morie.survey against the survey package on fixed data."""

from morie import survey as S
from morie.fn import _frame_core as pd

Y = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8, 4.2, 5.0]
X = [2.1, 2.9, 2.0, 3.3, 2.6, 3.0, 2.4, 2.7, 2.5, 3.1, 1.9, 2.2]
W = [10, 12, 8, 15, 9, 11, 14, 10, 13, 9, 12, 16]
PI = [0.1, 0.08, 0.12, 0.07, 0.11, 0.09, 0.07, 0.1, 0.08, 0.11, 0.09, 0.06]
D = pd.DataFrame(
    {
        "y": Y,
        "x": X,
        "w": W,
        "dom": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        "x2": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        "c": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "s": [1] * 6 + [2] * 6,
        "b": [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    }
)


def rel(a, b):
    return abs(a - b) / abs(b)


def test_estimator_ses_match_survey():
    # svymean, svyratio (SE x 70), svymean(subset(., dom == 1)), and svytotal
    # under poisson_sampling(pi), all with ids = ~1
    assert rel(S.hajek_mean(Y, W)["se"], 0.2735903604313098) <= 1e-13
    assert rel(S.ratio_estimator(Y, X, W, 70)["se"], 1.3338586248673094) <= 1e-13
    assert rel(S.subpopulation_estimate(D, "dom", 1, "y", "w")["se"], 0.35636988931716962) <= 1e-13
    assert rel(S.horvitz_thompson_total(Y, PI)["se"], 227.5546234351398) <= 1e-13


def test_raking_matches_survey_calibrate():
    # calibrate(d0, ~ -1 + x + x2, population = c(40, 7.5), calfun = "raking")
    w = S.calibration_weights(D, ["x", "x2"], {"x": 40, "x2": 7.5}, tol=1e-13).tolist()
    ref = [
        1.2170070612763382,
        1.3511035483545026,
        1.2044439471380985,
        1.4083631497992353,
        1.2818158022889989,
        1.3361389976162683,
        1.2827916116548395,
        1.3233527532730389,
        1.2685836701668838,
        1.3794362794509469,
        1.192010521513486,
        1.2564438996747997,
    ]
    assert max(rel(a, b) for a, b in zip(w, ref)) <= 1e-11
    assert abs(sum(a * b for a, b in zip(w, X)) - 40) <= 1e-10


def test_design_based_glm_matches_svyglm():
    r = S.complex_survey_glm(D, "y ~ x", "w")
    assert max(rel(a, b) for a, b in zip(r.params, [0.69876373111888657, 1.9631955182912122])) <= 1e-12
    assert max(rel(a, b) for a, b in zip(r.bse, [0.27154627726592373, 0.094242076051449811])) <= 1e-12
    r = S.complex_survey_glm(D, "y ~ x", "w", cluster_col="c")
    assert max(rel(a, b) for a, b in zip(r.bse, [0.30324142084729888, 0.1065966955724697])) <= 1e-12
    r = S.complex_survey_glm(D, "y ~ x", "w", cluster_col="c", strata_col="s")
    assert max(rel(a, b) for a, b in zip(r.bse, [0.24484209960379669, 0.089239305984671738])) <= 1e-12
    assert r.df_resid == 3
    # svyglm's SE here uses glm's penultimate-iterate working weights, so it
    # agrees to the IRLS tolerance rather than to rounding
    r = S.complex_survey_glm(D, "b ~ x", "w", family="binomial")
    assert max(rel(a, b) for a, b in zip(r.params, [-4.8866200346840696, 1.9936432258120707])) <= 1e-10
    assert max(rel(a, b) for a, b in zip(r.bse, [3.7328187074184713, 1.4190121127168933])) <= 1e-6
