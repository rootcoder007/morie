"""Gibbons cluster A: rank correlation and concordance.

gb1121 gb1122t gb1131n gb1131t gb1141 gb1241 gb1241t gb_kt2 gb_ktv
gb_sp2 gb_spv gb_wcin gb_blt. Oracles: scipy kendalltau/spearmanr,
exact enumeration, and the Gibbons closed forms (Ch 11-12,
PDF-verified eq. 12.4.4)."""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.gb1121 import gibbons_kendall_tau
from morie.fn.gb1122t import gibbons_kendall_ties
from morie.fn.gb1131n import gibbons_spearman_asymp
from morie.fn.gb1131t import gibbons_spearman_ties
from morie.fn.gb1141 import gibbons_tau_rho_relation
from morie.fn.gb1241 import gibbons_concordance_w
from morie.fn.gb1241t import gibbons_concordance_w_ties
from morie.fn.gb_blt import gibbons_balance_incomplete
from morie.fn.gb_kt2 import gibbons_kendall_exact
from morie.fn.gb_ktv import gibbons_kendall_tau_var
from morie.fn.gb_sp2 import gibbons_spearman_exact
from morie.fn.gb_spv import gibbons_spearman_rho_var
from morie.fn.gb_wcin import gibbons_concordance_incomplete


def test_kendall_tau_matches_scipy_on_tie_free_data():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(30)
        y = 0.5 * x + rng.standard_normal(30)
        ours = gibbons_kendall_tau(x, y)
        ref = stats.kendalltau(x, y)
        assert ours["tau"] == pytest.approx(ref.statistic, abs=1e-12)
    # perfect concordance and discordance hit the endpoints exactly
    assert gibbons_kendall_tau([1, 2, 3, 4], [10, 20, 30, 40])["tau"] == 1.0
    assert gibbons_kendall_tau([1, 2, 3, 4], [4, 3, 2, 1])["tau"] == -1.0
    with pytest.raises(ValueError):
        gibbons_kendall_tau([1.0], [2.0])


def test_tau_b_matches_scipy_with_ties_and_reduces_without():
    x = np.array([1, 2, 2, 3, 4, 4, 4, 5], dtype=float)
    y = np.array([2, 1, 3, 3, 5, 4, 6, 7], dtype=float)
    ours = gibbons_kendall_ties(x, y)
    ref = stats.kendalltau(x, y, variant="b")
    assert ours["tau_b"] == pytest.approx(ref.statistic, abs=1e-12)
    assert ours["T_x"] > 0
    # no ties: tau_b == plain tau
    a = np.array([3.0, 1.0, 4.0, 1.5, 5.0])
    b = np.array([2.0, 7.0, 1.0, 8.0, 2.5])
    assert gibbons_kendall_ties(a, b)["tau_b"] == pytest.approx(
        gibbons_kendall_tau(a, b)["tau"]
    )
    with pytest.raises(ValueError):
        gibbons_kendall_ties([1, 1, 1], [1, 1, 1])


def test_kendall_exact_enumeration_matches_the_closed_form_variance():
    for n in (4, 5, 6):
        out = gibbons_kendall_exact(n)
        assert out["mean"] == pytest.approx(0.0, abs=1e-12)
        assert out["var"] == pytest.approx(
            gibbons_kendall_tau_var(n)["var_tau"], abs=1e-12
        )
        assert np.sum(out["pmf"]) == pytest.approx(1.0)
        # symmetry of the null distribution
        assert out["support"] == pytest.approx(-out["support"][::-1])
    # exact p at the maximum: only the identity permutation
    import math

    out = gibbons_kendall_exact(5, t=1.0)
    assert out["p_ge"] == pytest.approx(1.0 / math.factorial(5))
    with pytest.raises(ValueError):
        gibbons_kendall_exact(12)


def test_spearman_exact_enumeration_variance_is_one_over_n_minus_1():
    for n in (4, 5, 6):
        out = gibbons_spearman_exact(n)
        assert out["mean"] == pytest.approx(0.0, abs=1e-12)
        assert out["var"] == pytest.approx(1.0 / (n - 1), abs=1e-12)  # exact
        assert gibbons_spearman_rho_var(n)["var"] == pytest.approx(out["var"])
    import math

    out = gibbons_spearman_exact(5, rho=1.0)
    assert out["p_ge"] == pytest.approx(1.0 / math.factorial(5))


def test_spearman_ties_equals_pearson_of_midranks():
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(40), 1)  # rounding forces ties
    y = np.round(0.6 * x + rng.standard_normal(40), 1)
    ours = gibbons_spearman_ties(x, y)
    ref = stats.spearmanr(x, y)
    assert ours["r_s"] == pytest.approx(ref.statistic, abs=1e-10)
    assert ours["T_x"] > 0
    with pytest.raises(ValueError):
        gibbons_spearman_ties([1, 1, 1], [1, 2, 3])


def test_spearman_asymp_and_relation_bounds():
    out = gibbons_spearman_asymp(0.5, 26)
    assert out["z"] == pytest.approx(0.5 * 5.0)
    assert out["p_two_sided"] == pytest.approx(2 * stats.norm.sf(2.5))
    assert gibbons_spearman_asymp(0.5, 8)["large_sample_ok"] is False
    # tau/rho bounds: an actual sample always satisfies them
    rng = np.random.default_rng(2)
    for _ in range(5):
        x = rng.standard_normal(25)
        y = 0.4 * x + rng.standard_normal(25)
        t = gibbons_kendall_tau(x, y)["tau"]
        r = stats.spearmanr(x, y).statistic
        assert gibbons_tau_rho_relation(t, r)["consistent"] is True
    # a pair violating Daniels is flagged
    assert gibbons_tau_rho_relation(0.9, -0.9)["consistent"] is False
    with pytest.raises(ValueError):
        gibbons_tau_rho_relation(1.5, 0.0)


def test_concordance_w_endpoints_and_chi2():
    # perfect agreement: W = 1, mean rho = 1
    R = np.tile(np.arange(1, 7), (4, 1))
    out = gibbons_concordance_w(R)
    assert out["W"] == pytest.approx(1.0)
    assert out["mean_spearman"] == pytest.approx(1.0)
    assert out["chi2"] == pytest.approx(4 * 5)
    # independent random rankings: W near 0, not significant
    rng = np.random.default_rng(3)
    R2 = np.array([rng.permutation(np.arange(1, 13)) + 0.0 for _ in range(3)])
    low = gibbons_concordance_w(R2)
    assert low["W"] < 0.5
    # ties: corrected W equals plain W when tie-free
    assert gibbons_concordance_w_ties(R)["W"] == pytest.approx(1.0)
    tied = np.array([[1.5, 1.5, 3, 4], [1, 2, 3, 4], [2, 1, 3.5, 3.5]])
    wt = gibbons_concordance_w_ties(tied)
    assert 0 < wt["W"] <= 1
    assert wt["tie_sum"] > 0
    with pytest.raises(ValueError):
        gibbons_concordance_w(np.arange(4.0))


def test_incomplete_concordance_agreement_beats_disagreement():
    nan = np.nan
    # judges agree on overlapping subsets
    agree = np.array([
        [1, 2, 3, nan],
        [nan, 1, 2, 3],
        [1, 2, nan, 3],
    ])
    disagree = np.array([
        [3, 2, 1, nan],
        [nan, 3, 2, 1],
        [1, 2, nan, 3],
    ])
    wa = gibbons_concordance_incomplete(agree)
    wd = gibbons_concordance_incomplete(disagree)
    assert wa["W"] > wd["W"]
    assert 0 <= wd["W"] <= 1
    with pytest.raises(ValueError):
        gibbons_concordance_incomplete(np.array([[1, nan], [2, nan]]))  # object never ranked


def test_bib_concordance_validates_the_design():
    nan = np.nan
    # symmetric BIB: n = 4 objects, b = 4 blocks of m = 3, r = 3, lam = 2
    bib_perfect = np.array([
        [1, 2, 3, nan],
        [1, 2, nan, 3],
        [1, nan, 2, 3],
        [nan, 1, 2, 3],
    ])
    out = gibbons_balance_incomplete(bib_perfect)
    assert out["lambda_"] == 2
    assert out["m_per_block"] == 3
    assert out["r_per_object"] == 3
    assert 0 < out["W_b"] <= 1
    # scrambled blocks give lower W_b
    bib_noise = np.array([
        [3, 1, 2, nan],
        [2, 3, nan, 1],
        [1, nan, 3, 2],
        [nan, 3, 1, 2],
    ])
    assert gibbons_balance_incomplete(bib_noise)["W_b"] < out["W_b"]
    # a non-BIB layout raises instead of being scored with BIB constants
    bad = np.array([
        [1, 2, 3, nan],
        [1, 2, 3, nan],
        [1, nan, 2, 3],
        [nan, 1, 2, 3],
    ])
    with pytest.raises(ValueError):
        gibbons_balance_incomplete(bad)
