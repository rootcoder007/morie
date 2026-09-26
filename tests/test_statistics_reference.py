"""morie.statistics against base R, nortest, irr and psych on fixed data."""

from morie import statistics as S

X = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8]
Y = [4.2, 5.0, 3.9, 5.8, 4.4, 4.9, 5.3, 4.1, 4.6, 5.2]
Z = [6.1, 5.2, 6.9, 7.4, 6.0, 5.8, 7.1, 6.6, 6.2, 6.4]


def rel(a, b):
    return abs(a - b) / abs(b)


def test_rank_tests_exact_p():
    # cor.test(method = "spearman") exact (AS 89) p; wilcox.test exact p
    assert rel(S.spearman_correlation(X, Z).p_value, 0.86475352880452594) <= 1e-12
    mw = S.mann_whitney_u(X, Y)
    assert rel(mw.p_value, 0.0015046872632011952) <= 1e-12
    assert mw.effect_size == 2 * 90 / 100 - 1


def test_mcnemar_exact_statistic_is_discordant_count():
    # binom.test(5, 18)
    r = S.mcnemar_test([[12, 5], [13, 9]], exact=True)
    assert r.test_statistic == 5.0
    assert rel(r.p_value, 0.096252441406250014) <= 1e-12


def test_normality_p_values_match_nortest():
    assert rel(S.anderson_darling(X).p_value, 0.96933659577224474) <= 1e-12
    lf = S.lilliefors_test(X)
    assert rel(lf.test_statistic, 0.1239910911322597) <= 1e-12
    assert rel(lf.p_value, 0.93376194173627036) <= 1e-12
    assert rel(S.dagostino_pearson(X + Y).test_statistic, 0.745242299228) <= 1e-10


def test_kappa_fleiss_cohen_everitt_variances():
    # irr::kappa2 z and p; psych::cohen.kappa interval
    r = S.cohens_kappa([1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 3, 2], [1, 2, 3, 2, 2, 3, 1, 1, 3, 3, 2, 2])
    assert abs(r.estimate - 0.625) <= 1e-14
    assert rel(r.test_statistic, 3.09426373877638) <= 1e-12
    assert rel(r.p_value, 0.00197302008751898) <= 1e-10
    assert abs(r.ci_lower - 0.2568693) <= 1e-6


def test_fisher_conditional_and_chi2_on_lists():
    fe = S.fisher_exact_test([[12, 5], [3, 9]])
    assert rel(fe.estimate, 6.654201) <= 1e-4 and rel(fe.ci_lower, 1.076137) <= 1e-4
    assert fe.extra["sample_odds_ratio"] == 7.2
    ci = S.chi2_independence([[12, 5, 7], [3, 9, 6], [8, 4, 10]])
    assert rel(ci.test_statistic, 8.3659786268481913) <= 1e-12


def test_icc_rm_fleiss_partial_twoway_match_references():
    # psych::ICC(Y, lmer = FALSE), car::Anova (type II; idata GG), irr::kappam.fleiss, ppcor::pcor.test
    import math

    from morie.fn import _frame_core as pd

    M = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    long = pd.DataFrame(
        {
            "t": [i for i in range(6) for _ in range(4)],
            "r": [j for _ in range(6) for j in range(4)],
            "v": [float(M[i][j]) for i in range(6) for j in range(4)],
        }
    )
    icc = {
        "ICC1": 0.16574176840547522,
        "ICC2": 0.28976377952755894,
        "ICC3": 0.71484071484071465,
        "ICC1k": 0.44279713367926826,
        "ICC2k": 0.62005054759898892,
        "ICC3k": 0.90931554237706935,
    }
    lo = {
        "ICC1": -0.1329323248747511,
        "ICC2": 0.018786513374711964,
        "ICC3": 0.34246476503392492,
        "ICC1k": -0.8844421552381212,
        "ICC2k": 0.071136815302503181,
        "ICC3k": 0.67567471381630428,
    }
    for t, v in icc.items():
        r = S.intraclass_correlation(long, "t", "r", "v", t)
        assert rel(r.estimate, v) <= 1e-12
        assert rel(r.ci_lower, lo[t]) <= 1e-10
    rm = S.repeated_measures_anova(long, "v", "t", "r")
    assert rel(rm.test_statistic, 31.86648501362399) <= 1e-12
    assert rel(rm.extra["epsilon_gg"], 0.68172454459409559) <= 1e-12
    assert rel(rm.p_value, 3.8455263546962374e-05) <= 1e-10
    fm = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    fk = S.fleiss_kappa(fm)
    assert rel(fk.estimate, 0.20993070442195522) <= 1e-12
    assert rel(fk.test_statistic, 12.374291059190464) <= 1e-12
    z = [[a, b] for a, b in zip(Z, [1.0, 3.0, 2.0, 5.0, 4.0, 2.5, 3.5, 1.5, 4.5, 3.2])]
    pc = S.partial_correlation(X, Y, z)
    assert rel(pc.estimate, 0.67284450504527471) <= 1e-12
    assert rel(pc.p_value, 0.067465424202641774) <= 1e-10
    sp = S.semi_partial_correlation(X, Y, z)
    r = sp.estimate
    t = r * math.sqrt(6 / (1 - r * r))
    from morie.fn import _stats_core as st

    assert rel(sp.p_value, 2 * st.t.sf(abs(t), 6)) <= 1e-12
    d2 = pd.DataFrame({"y": X + Y + Z, "a": ["p"] * 13 + ["q"] * 17, "b": ["u", "v", "w"] * 10})
    tab = S.two_way_anova(d2, "y", "a", "b").extra["anova_table"]
    ss = [tab["sum_sq"][k] for k in ("C(a)", "C(b)", "C(a):C(b)", "Residual")]
    for a, b in zip(ss, [0.18116438356164366, 0.18021868220417758, 0.57866894977168926, 25.92916666666666]):
        assert rel(a, b) <= 1e-12
    pv = [tab["PR(>F)"][k] for k in ("C(a)", "C(b)", "C(a):C(b)")]
    for a, b in zip(pv, [0.68580884107275075, 0.92024387320657997, 0.76731111013819064]):
        assert rel(a, b) <= 1e-10
