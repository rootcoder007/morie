"""Tests for the Ghosal Ch 3 construction modules."""
from morie.fn.gh_c3_1 import ghosal_random_measure_def
from morie.fn.gh_c3_2 import ghosal_stochastic_proc_prior
from morie.fn.gh_c3_3 import ghosal_dir_simplex
from morie.fn.gh_c3_4 import ghosal_stick_break_def
from morie.fn.gh_c3_5 import ghosal_countable_dp
from morie.fn.gh_c3_6 import ghosal_dense_subset_prior
from morie.fn.gh_c3_7 import ghosal_rect_partition
from morie.fn.gh_c3_9 import ghosal_quantile_prior
from morie.fn.gh_c3_10 import ghosal_norm_crm
from morie.fn.gh_c3_11 import ghosal_tailfree_def
from morie.fn.gh_c3_12 import ghosal_polya_tree_def
from morie.fn.gh_c3_13 import ghosal_polya_urn_pt
from morie.fn.gh_c3_15 import ghosal_partspec_pt
from morie.fn.gh_c3_16 import ghosal_evsplit_pt

X = [0.3]


def test_random_measure_additive_and_normalized():
    r = ghosal_random_measure_def(X)
    assert r["additivity_gap"] < 1e-12
    assert abs(r["total_mass"] - 1.0) < 1e-12


def test_consistency_by_aggregation():
    r = ghosal_stochastic_proc_prior(X)
    assert r["aggregation_gap"] < 1e-12
    assert abs(sum(r["weights"]) - 1.0) < 1e-12


def test_dirichlet_simplex():
    r = ghosal_dir_simplex(X, alpha=[2.0, 2.0, 2.0], seed=7)
    assert abs(sum(r["p"]) - 1.0) < 1e-12
    assert min(r["p"]) > 0


def test_stick_breaking_mass_near_one():
    r = ghosal_stick_break_def(X, n_terms=200)
    assert 0.999 < r["total_mass"] <= 1.0 + 1e-12


def test_countable_dp_cells_plus_tail():
    r = ghosal_countable_dp(X)
    assert abs(sum(r["p_cells"]) + r["p_tail"] - 1.0) < 1e-12


def test_dense_atoms_are_dyadic():
    r = ghosal_dense_subset_prior(X)
    assert r["atoms"][:3] == [0.5, 0.25, 0.75]


def test_rect_partition_mass_conserved():
    r = ghosal_rect_partition(X, depth=6)
    assert abs(r["total_mass"] - 1.0) < 1e-12
    assert r["n_cells"] == 64


def test_quantile_prior_monotone():
    assert ghosal_quantile_prior(X)["monotone"] is True


def test_crm_normalizes():
    r = ghosal_norm_crm(X)
    assert 0.2 < r["estimate"] < 0.8


def test_tailfree_prop312_mean():
    # Prop 3.12(i): E P(leftmost level-m cell) = 2^-m under
    # symmetric Beta splits; Monte Carlo gap stays small
    r = ghosal_tailfree_def(X, depth=6, seed=1)
    assert r["prop312_gap"] < 0.02


def test_polya_tree_positive_density():
    r = ghosal_polya_tree_def(X, seed=2)
    assert r["estimate"] > 0


def test_urn_predictive_exact():
    # 3 of 4 obs in B, alpha_B=1, alpha=2: (1+3)/(2+4) = 2/3
    r = ghosal_polya_urn_pt([0.1, 0.2, 0.4, 0.9])
    assert abs(r["estimate"] - 4.0 / 6.0) < 1e-12


def test_partspec_only_specified_levels_random():
    a = ghosal_partspec_pt(X, levels=(2, 4), seed=3)
    assert a["cell_mass"] > 0


def test_evsplit_prior_uniform_posterior_concentrates():
    prior = ghosal_evsplit_pt([0.3])
    assert abs(prior["estimate"] - 1.0) < 1e-12
    post = ghosal_evsplit_pt([0.3], data=[0.3] * 40)
    far = ghosal_evsplit_pt([0.9], data=[0.3] * 40)
    assert post["estimate"] > 1.0 > far["estimate"]
