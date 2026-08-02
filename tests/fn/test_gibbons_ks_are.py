"""Gibbons clusters D+E+F: K-S, tie corrections, linear rank, ARE,
fundamentals, association. Oracles: scipy (kstwobign, mannwhitneyu,
kruskal, chi2_contingency), exact enumeration, and the PDF-verified
Table 13.3.1 / Sec 13.3.3 constants."""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.gb433 import gibbons_ks_kolmogorov_limit
from morie.fn.gb434bt import gibbons_ks_bt_formula
from morie.fn.gb435 import gibbons_ks_onesided_asymp
from morie.fn.gb4351 import gibbons_ks_chi2_approx
from morie.fn.gb_pp import gibbons_pp_plot
from morie.fn.gb_qq import gibbons_qq_plot
from morie.fn.gb661v import gibbons_mw_var
from morie.fn.gb661t import gibbons_mw_ties
from morie.fn.gb821t import gibbons_wrs_ties
from morie.fn.gb1041t import gibbons_kw_ties
from morie.fn.gb_binmw import gibbons_mw_binomial_link
from morie.fn.gb_mw2 import gibbons_mw_rs_equiv
from morie.fn.gb734 import gibbons_linrank_symmetry_cond
from morie.fn.gb735 import gibbons_linrank_sym_equal
from morie.fn.gb736 import gibbons_linrank_sym_special
from morie.fn.gb7381 import gibbons_cs_null_var
from morie.fn.gb1321 import gibbons_are_def
from morie.fn.gb1323 import gibbons_are_twosided
from morie.fn.gb_are1 import gibbons_are_sign_wilcoxon
from morie.fn.gb_are2 import gibbons_are_normal_case
from morie.fn.gb_are3 import gibbons_are_dbl_exp
from morie.fn.gb_are4 import gibbons_are_kw
from morie.fn.gb_are5 import gibbons_are_scale_tests
from morie.fn.gb_ar6 import gibbons_are_unif
from morie.fn.gb_ar7 import gibbons_are_logistic
from morie.fn.gb_psi import gibbons_pitman_efficiency
from morie.fn.gb_c1 import gibbons_chebyshev
from morie.fn.gb_c2 import gibbons_chi2_yates
from morie.fn.gb_cc import gibbons_continuity_corr
from morie.fn.gb_clt import gibbons_clt
from morie.fn.gb1421c import gibbons_contingency_coeff
from morie.fn.gb1421t import gibbons_phi_cramers_v
from morie.fn.gb_cq import gibbons_cramers_contingency


def test_kolmogorov_limit_matches_scipy_kstwobign():
    for d in (0.5, 0.8, 1.0, 1.36, 2.0):
        out = gibbons_ks_kolmogorov_limit(d)
        assert out["L"] == pytest.approx(stats.kstwobign.cdf(d), abs=1e-10)
    # the classical 5% point: L(1.3581) = 0.95
    assert gibbons_ks_kolmogorov_limit(1.3581015)["L"] == pytest.approx(0.95, abs=1e-4)
    with pytest.raises(ValueError):
        gibbons_ks_kolmogorov_limit(-1.0)


def test_birnbaum_tingey_exact_vs_scipy_and_the_asymptotics():
    # exact one-sided tail vs scipy's ksone survival function
    for n in (10, 25):
        for c in (0.15, 0.25, 0.4):
            out = gibbons_ks_bt_formula(c, n)
            assert out["p_exceed"] == pytest.approx(
                stats.ksone.sf(c, n), abs=1e-10
            )
    # Theorem 4.3.5 limit approached from the exact formula
    d = 1.0
    exact_1000 = gibbons_ks_bt_formula(d / np.sqrt(1000), 1000)["p_exceed"]
    assert exact_1000 == pytest.approx(np.exp(-2 * d**2), abs=0.01)
    assert gibbons_ks_onesided_asymp(d)["p_value"] == pytest.approx(np.exp(-2.0))
    # chi-square clothing: identical p-values
    chi = gibbons_ks_chi2_approx(400, 0.05)
    assert chi["p_value"] == pytest.approx(np.exp(-2 * 400 * 0.05**2), abs=1e-12)
    with pytest.raises(ValueError):
        gibbons_ks_bt_formula(1.5, 10)


def test_pp_and_qq_plots_diagnose_fit():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    pp = gibbons_pp_plot(x)
    # correct null: departure small; the departure IS the K-S D_n
    assert pp["max_departure"] == pytest.approx(
        stats.kstest(x, "norm").statistic, abs=1e-12
    )
    qq = gibbons_qq_plot(3.0 + 2.0 * x)
    assert qq["correlation"] > 0.99
    assert qq["slope"] == pytest.approx(2.0, abs=0.2)  # scale recovered
    assert qq["intercept"] == pytest.approx(3.0, abs=0.2)  # location recovered
    # wrong family: exponential data on a normal Q-Q bends
    assert gibbons_qq_plot(rng.exponential(size=200))["correlation"] < \
        gibbons_qq_plot(rng.standard_normal(200))["correlation"]


def test_tie_corrected_two_sample_variances():
    assert gibbons_mw_var(6, 8)["var"] == pytest.approx(6 * 8 * 15 / 12)
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(30), 0)  # heavy ties
    y = np.round(rng.standard_normal(35) + 0.3, 0)
    mw = gibbons_mw_ties(x, y)
    assert mw["var_corrected"] < mw["var_uncorrected"]  # ties always shrink
    # z matches scipy's tie-corrected normal method
    ref = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic",
                             use_continuity=False)
    assert mw["p_two_sided"] == pytest.approx(ref.pvalue, abs=1e-10)
    # W and U carry the same tie-corrected variance
    wrs = gibbons_wrs_ties(x, y)
    assert wrs["var_corrected"] == pytest.approx(mw["var_corrected"])
    assert abs(wrs["z"]) == pytest.approx(abs(mw["z"]), abs=1e-12)
    # KW correction only raises H, and matches scipy
    g = [np.round(rng.standard_normal(20) + d, 0) for d in (0.0, 0.4, 0.8)]
    kw = gibbons_kw_ties(g)
    assert kw["H"] >= kw["H_uncorrected"]
    ref_kw = stats.kruskal(*g)
    assert kw["H"] == pytest.approx(ref_kw.statistic, abs=1e-10)
    with pytest.raises(ValueError):
        gibbons_kw_ties([g[0]])


def test_w_u_linkage_and_rank_pair_equivalence():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal(12), rng.standard_normal(15)
    eq = gibbons_mw_rs_equiv(x, y)
    assert eq["identity_holds"] is True
    link = gibbons_mw_binomial_link(eq["W"], 12)
    assert link["U"] == pytest.approx(eq["U_direct"])
    with pytest.raises(ValueError):
        gibbons_mw_binomial_link(5.0, 12)  # below the minimum rank sum


def test_linear_rank_symmetry_theorems_by_enumeration():
    # 7.3.4: Wilcoxon scores a_i = i are complementary (c = N + 1)
    out = gibbons_linrank_symmetry_cond(np.arange(1, 9))
    assert out["symmetric"] is True and out["constant"] == 9.0
    assert gibbons_linrank_symmetry_cond([1.0, 5.0, 2.0])["symmetric"] is False
    # 7.3.5: m = n makes ANY scores symmetric -- even ugly ones
    ugly = np.array([0.0, 1.0, 1.5, 7.0, 7.2, 11.0])
    assert gibbons_linrank_sym_equal(ugly, 3, 3)["symmetric"] is True
    # and unequal sizes with ugly scores are NOT symmetric
    assert gibbons_linrank_sym_equal(ugly, 2, 4)["symmetric"] is False
    # 7.3.6: folded scores symmetric at any split -- for EVEN N only.
    # The book's proof swaps the two halves (Z'_i = Z_{i+N/2}, p. 282),
    # which needs N/2 integral; enumeration shows N = 7 is skewed.
    sc = gibbons_linrank_sym_special(8)
    assert sc["symmetric"] is True and sc["palindromic"] is True
    assert gibbons_linrank_sym_equal(sc["scores"], 3, 5)["symmetric"] is True
    assert gibbons_linrank_sym_equal(sc["scores"], 2, 6)["symmetric"] is True
    with pytest.raises(ValueError):
        gibbons_linrank_sym_special(7)  # odd N: genuinely skewed
    # 7.3.1 Chernoff-Savage: Wilcoxon J(u) = u gives Var J(U) = 1/12
    cs = gibbons_cs_null_var(lambda u: u, 0.5)
    assert cs["var_J"] == pytest.approx(1.0 / 12.0, abs=1e-10)
    assert cs["limit"] == pytest.approx(0.5 / 12.0, abs=1e-10)
    with pytest.raises(ValueError):
        gibbons_cs_null_var(lambda u: u, 1.5)


def test_are_table_13_3_1_and_the_efficacy_rederivation():
    # PDF-verified Table 13.3.1
    nrm = gibbons_are_normal_case()
    assert nrm["wilcoxon_vs_t"] == pytest.approx(3 / np.pi)
    assert nrm["sign_vs_t"] == pytest.approx(2 / np.pi)
    assert nrm["sign_vs_wilcoxon"] == pytest.approx(2 / 3)
    de = gibbons_are_dbl_exp()
    assert de["wilcoxon_vs_t"] == pytest.approx(1.5)
    assert de["sign_vs_t"] == pytest.approx(2.0)
    assert gibbons_are_unif()["wilcoxon_vs_t"] == pytest.approx(1.0)
    assert gibbons_are_logistic()["wilcoxon_vs_t"] == pytest.approx(np.pi**2 / 9)
    # the efficacy integrals REPRODUCE the table -- for every family
    for mod in (nrm, de, gibbons_are_unif(), gibbons_are_logistic()):
        for k, v in mod["derived"].items():
            assert v == pytest.approx(mod[k], rel=1e-6), (mod["distribution"], k)
    # every family sits above the Hodges-Lehmann bound
    for d in ("uniform", "normal", "logistic", "double_exponential"):
        assert gibbons_are_kw(d)["above_bound"] is True
    # scale tests: the PDF-verified 15/(2 pi^2), NOT the placeholder's 3/pi
    sc = gibbons_are_scale_tests()
    assert sc["are_mood_f"] == pytest.approx(15 / (2 * np.pi**2))
    assert sc["are_mood_f"] != pytest.approx(3 / np.pi, abs=0.01)
    assert sc["are_klotz_f"] == 1.0
    # general-density module agrees with the table at the normal
    g = gibbons_are_sign_wilcoxon(stats.norm.pdf)
    assert g["are"] == pytest.approx(2 / 3, rel=1e-6)
    # ... and is scale-free
    g2 = gibbons_are_sign_wilcoxon(lambda x: stats.norm.pdf(x, scale=3) )
    assert g2["are"] == pytest.approx(g["are"], rel=1e-6)


def test_are_definition_and_two_sided_invariance():
    out = gibbons_are_def(1.0, 2.0, n=100)
    assert out["are"] == pytest.approx(0.25)
    assert out["n_star_for_equal_power"] == pytest.approx(25.0)
    two = gibbons_are_twosided(1.0, 2.0)
    assert two["are_two_sided"] == two["are_one_sided"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        gibbons_are_def(-1.0, 2.0)
    # simulated Pitman efficiency: t vs sign test on normal data --
    # the ratio should favour the t (ratio < 1 for sign vs t ordering)
    def t_test(x):
        return stats.ttest_1samp(x, 0.0).pvalue

    def sign_test(x):
        k = int(np.sum(x > 0))
        return stats.binomtest(k, x.size, 0.5).pvalue

    sim = gibbons_pitman_efficiency(
        sign_test, t_test, lambda th, n, rng: rng.standard_normal(n) + th,
        delta=0.3, n=60, n_sim=300,
    )
    assert sim["power2"] >= sim["power1"]  # t at least as powerful at the normal


def test_fundamentals_and_association():
    # Chebyshev: bound holds empirically, with visible slack at the normal
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000)
    ch = gibbons_chebyshev(2.0, x)
    assert ch["empirical"] <= ch["bound"]
    assert ch["empirical"] < 0.06  # normal true value ~0.0455 << 0.25
    # CLT z
    clt = gibbons_clt(xbar=0.2, n=100, mu=0.0, sigma=1.0)
    assert clt["z"] == pytest.approx(2.0)
    # continuity correction weakens, never strengthens
    cc = gibbons_continuity_corr(60, 50, 5.0)
    assert cc["z_corrected"] < abs(cc["z_uncorrected"])
    assert cc["p_two_sided"] > cc["p_uncorrected"]
    # Yates matches scipy's corrected chi2 on a 2x2
    tbl = [[18, 7], [6, 19]]
    ya = gibbons_chi2_yates(tbl)
    ref = stats.chi2_contingency(tbl, correction=True)
    assert ya["chi2_corrected"] == pytest.approx(ref.statistic, abs=1e-10)
    with pytest.raises(ValueError):
        gibbons_chi2_yates(np.ones((3, 3)))
    # phi == V on 2x2; V and C move together; C < C_max
    pv = gibbons_phi_cramers_v(tbl)
    assert pv["phi"] == pytest.approx(pv["cramers_v"])
    assert gibbons_cramers_contingency(tbl)["cramers_v"] == pytest.approx(
        pv["cramers_v"]
    )
    cco = gibbons_contingency_coeff(tbl)
    assert 0 < cco["C"] < cco["C_max"] < 1
    assert cco["C_adjusted"] > cco["C"]
    # independence: everything near zero
    ind = [[25, 25], [25, 25]]
    assert gibbons_phi_cramers_v(ind)["cramers_v"] == pytest.approx(0.0)
