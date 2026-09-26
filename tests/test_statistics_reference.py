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
