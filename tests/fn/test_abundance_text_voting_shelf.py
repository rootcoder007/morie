"""Abundance, loss variance, text confounding, covariance, DAGs, voting.

Sources: Lu et al (2017) *PeerJ CS* 3:e104 (Bracken); Silver et al
(2018) *Science* 362:1140-1144 and Geyer (1992) *Stat Sci* 7:473-483;
Veitch, Sridhar and Blei (2020) arXiv:1905.12741; Xiao et al (2016)
*Stat Comput* 26:409-421; Pearl (2009) *Causality* Sec 3.3; Poole and
Rosenthal (1985) *AJPS* 29:357-384.
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn.abndst import (
    abundance_estimation,
    kmer_distribution_from_assignments,
)
from morie.fn.aglnvr import alphazero_loss_var, effective_sample_size
from morie.fn.cbnrt import causalbert_text, text_embedding, tfidf_matrix
from morie.fn.facea import bspline_basis, face_smooth
from morie.fn.midor import (
    backdoor_sets,
    is_backdoor_admissible,
    model_identify_estimate_refute,
)
from morie.fn.wnoma import wnominate_alternating


# --------------------------------------------------------------------
# Bracken abundance
# --------------------------------------------------------------------

def test_shared_ancestor_reads_split_by_kmer_compatibility():
    P = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    out = abundance_estimation(np.array([100.0, 300.0, 400.0]), P)
    assert out["estimate"] == pytest.approx([200.0, 600.0], rel=1e-6)
    assert out["converged"] is True


def test_the_naive_species_count_understates_and_by_unequal_amounts():
    # species 0 shares more genome, so more of its reads strand higher
    P = np.array([[0.3, 0.0], [0.0, 0.8], [0.7, 0.2]])
    reads = np.array([300.0, 800.0, 900.0])
    out = abundance_estimation(reads, P)
    naive = out["naive_species_reads"]
    assert naive[0] < out["estimate"][0]
    assert naive[1] < out["estimate"][1]
    # the shortfall differs between species, so the RANKING can flip
    assert abs(out["estimate"][0] - naive[0]) != pytest.approx(
        abs(out["estimate"][1] - naive[1]), rel=0.1
    )


def test_the_estimator_recovers_a_known_mixture():
    rng = np.random.default_rng(0)
    truth = np.array([0.5, 0.3, 0.2])
    P = np.array([
        [0.6, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.4],
        [0.4, 0.5, 0.0],
        [0.0, 0.0, 0.6],
    ])
    node_p = P @ truth
    reads = rng.multinomial(200000, node_p / node_p.sum()).astype(float)
    out = abundance_estimation(reads, P)
    assert out["fractions"] == pytest.approx(truth, abs=0.01)


def test_total_reads_are_conserved():
    P = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    reads = np.array([100.0, 300.0, 400.0])
    out = abundance_estimation(reads, P)
    assert float(np.sum(out["estimate"])) == pytest.approx(float(reads.sum()))


def test_indistinguishable_species_are_reported_not_split_silently():
    # identical columns leave the likelihood flat along their trade-off
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    out = abundance_estimation(np.array([100.0, 100.0]), P)
    assert out["identifiable"] is False
    assert out["max_column_cosine"] == pytest.approx(1.0)
    assert any("indistinguishable" in w for w in out.warnings)


def test_distinguishable_species_are_not_flagged():
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = abundance_estimation(np.array([100.0, 100.0]), P)
    assert out["identifiable"] is True
    assert not any("indistinguishable" in w for w in out.warnings)


def test_unreachable_nodes_are_excluded_rather_than_spread():
    P = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    out = abundance_estimation(np.array([100.0, 100.0, 50.0]), P)
    assert out["reads_unassignable"] == 50.0
    assert float(np.sum(out["estimate"])) == pytest.approx(200.0)
    assert any("no species" in w for w in out.warnings)


def test_kmer_distribution_normalises_columns():
    A = np.array([[3.0, 0.0], [1.0, 4.0]])
    P = kmer_distribution_from_assignments(A)
    assert P.sum(axis=0) == pytest.approx([1.0, 1.0])


def test_abundance_input_validation():
    P = np.array([[0.5, 0.0], [0.5, 1.0]])
    with pytest.raises(ValueError, match="must sum to 1"):
        abundance_estimation([1.0, 1.0], np.array([[0.3, 0.0], [0.3, 0.5]]))
    with pytest.raises(ValueError, match="non-negative"):
        abundance_estimation([-1.0, 1.0], P)
    with pytest.raises(ValueError, match="no reads"):
        abundance_estimation([0.0, 0.0], P)
    with pytest.raises(ValueError, match="undefined"):
        kmer_distribution_from_assignments(np.array([[1.0, 0.0],
                                                     [1.0, 0.0]]))


# --------------------------------------------------------------------
# Training-loss variance
# --------------------------------------------------------------------

def ar1(n, rho, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=sd, size=n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_an_independent_stream_needs_no_correction():
    rng = np.random.default_rng(0)
    out = alphazero_loss_var(rng.normal(2.0, 0.5, size=4000))
    assert out["estimate"] == pytest.approx(2.0, abs=0.05)
    assert out["se_inflation"] == pytest.approx(1.0, abs=0.25)


def test_autocorrelation_inflates_the_standard_error():
    # the naive s/sqrt(n) is the number people quote and it is too small
    out = alphazero_loss_var(2.0 + ar1(4000, 0.9, seed=1))
    assert out["se_inflation"] > 3.0
    assert out["se"] > out["se_naive"]
    assert any("autocorrelated" in w for w in out.warnings)


def test_the_inflation_matches_the_theoretical_ar1_factor():
    # for an AR(1), tau_int = (1 + rho) / (1 - rho)
    for rho in (0.5, 0.8):
        out = alphazero_loss_var(ar1(20000, rho, seed=2))
        theory = math.sqrt((1 + rho) / (1 - rho))
        assert out["se_inflation"] == pytest.approx(theory, rel=0.25)


def test_the_effective_sample_size_falls_as_correlation_rises():
    a = alphazero_loss_var(ar1(4000, 0.0, seed=3))["ess"]
    b = alphazero_loss_var(ar1(4000, 0.9, seed=3))["ess"]
    assert b < a / 3


def test_the_honest_interval_is_wider_than_the_naive_one():
    out = alphazero_loss_var(2.0 + ar1(4000, 0.85, seed=4))
    assert (out["ci_upper"] - out["ci_lower"]) > \
           (out["ci_naive_upper"] - out["ci_naive_lower"])


def test_shuffling_destroys_the_structure_being_measured():
    x = ar1(4000, 0.9, seed=5)
    rng = np.random.default_rng(0)
    ordered = alphazero_loss_var(x)["se_inflation"]
    shuffled = alphazero_loss_var(rng.permutation(x))["se_inflation"]
    assert ordered > 3.0
    assert shuffled == pytest.approx(1.0, abs=0.3)


def test_components_are_decomposed_and_shares_sum_to_one():
    rng = np.random.default_rng(6)
    n = 500
    v = rng.normal(1.0, 0.2, size=n)
    p = rng.normal(0.5, 0.1, size=n)
    r = rng.normal(0.1, 0.01, size=n)
    out = alphazero_loss_var(v + p + r, value_loss=v, policy_loss=p,
                             reg_loss=r)
    assert set(out["components"]) == {"value", "policy", "regularisation"}
    assert sum(out["component_shares"].values()) == pytest.approx(1.0,
                                                                 abs=1e-9)


def test_effective_sample_size_of_white_noise_is_about_n():
    rng = np.random.default_rng(7)
    ess = effective_sample_size(rng.normal(size=5000))["ess"]
    assert 3000 < ess < 8000


def test_loss_variance_input_validation():
    with pytest.raises(ValueError, match="at least one finite"):
        alphazero_loss_var([])
    with pytest.raises(ValueError, match="alpha"):
        alphazero_loss_var([1.0, 2.0], alpha=1.5)
    with pytest.raises(ValueError, match="value_loss has length"):
        alphazero_loss_var([1.0, 2.0], value_loss=[1.0])


# --------------------------------------------------------------------
# Text-borne confounding
# --------------------------------------------------------------------

def text_confounded(n=600, seed=0, tau=1.0):
    rng = np.random.default_rng(seed)
    sev = rng.random(n) < 0.4
    stem = ["routine case stable outcome",
            "severe acute presentation critical"]
    texts = [stem[int(s)] + " " + " ".join(
        rng.choice(["alpha", "beta", "gamma", "delta"], 4)) for s in sev]
    p = np.where(sev, 0.8, 0.25)
    T = (rng.random(n) < p).astype(float)
    Y = tau * T + 3.0 * sev + rng.normal(size=n)
    return texts, T, Y, sev


def test_adjusting_for_text_removes_the_confounding():
    texts, T, Y, _ = text_confounded()
    out = causalbert_text(texts, T, Y, n_components=5)
    assert abs(out["naive_difference"] - 1.0) > 1.0     # badly confounded
    assert out["estimate"] == pytest.approx(1.0, abs=0.2)


def test_the_movement_from_the_unadjusted_contrast_is_reported():
    texts, T, Y, _ = text_confounded()
    out = causalbert_text(texts, T, Y, n_components=5)
    assert out["adjustment_movement"] == pytest.approx(
        out["estimate"] - out["naive_difference"]
    )
    assert out["adjustment_movement"] < -1.0


def test_a_supplied_embedding_is_used_instead_of_the_bag_of_words():
    # this is the route to a pretrained representation: the function
    # consumes a matrix and takes no model dependency
    _, T, Y, sev = text_confounded()
    rng = np.random.default_rng(1)
    E = np.column_stack([sev.astype(float) + rng.normal(scale=0.1,
                                                        size=T.size),
                         rng.normal(size=T.size)])
    out = causalbert_text(None, T, Y, embedding=E)
    assert out["estimate"] == pytest.approx(1.0, abs=0.25)
    assert out["n_components"] == 2


def test_the_bag_of_words_cannot_see_negation():
    # the documented failure, measured rather than asserted
    A = tfidf_matrix(["history of psychosis", "no history of psychosis"])
    cos = float(A["matrix"][0] @ A["matrix"][1])
    assert cos > 0.7
    # once "no" falls below the document-frequency floor they coincide
    B = tfidf_matrix(["history of psychosis", "no history of psychosis"],
                     min_df=2)
    assert float(B["matrix"][0] @ B["matrix"][1]) == pytest.approx(1.0)
    assert "no" not in B["vocabulary"]


def test_tfidf_rows_are_unit_norm():
    A = tfidf_matrix(["a b c", "b c d d", "c"])["matrix"]
    assert np.sqrt((A ** 2).sum(axis=1)) == pytest.approx(np.ones(3))


def test_rare_terms_carry_more_weight_than_common_ones():
    out = tfidf_matrix(["common rare1", "common x", "common y", "common z"])
    v = out["vocabulary"]
    assert out["idf"][v.index("rare1")] > out["idf"][v.index("common")]


def test_the_embedding_explains_decreasing_variance():
    texts, _, _, _ = text_confounded()
    e = text_embedding(texts, n_components=5)
    evr = e["explained_variance_ratio"]
    assert np.all(np.diff(evr) <= 1e-12)
    assert 0 < float(np.sum(evr)) <= 1.0 + 1e-9


def test_the_bag_of_words_caveat_is_always_surfaced():
    texts, T, Y, _ = text_confounded(n=200)
    out = causalbert_text(texts, T, Y, n_components=4)
    assert any("negation" in w for w in out.warnings)


def test_text_confounding_input_validation():
    texts, T, Y, _ = text_confounded(n=200)
    with pytest.raises(ValueError, match="T must be binary"):
        causalbert_text(texts, T * 2, Y, n_components=3)
    with pytest.raises(ValueError, match="Y has length"):
        causalbert_text(texts, T, Y[:50], n_components=3)
    with pytest.raises(ValueError, match="documents but there are"):
        causalbert_text(texts[:10], T, Y, n_components=3)


# --------------------------------------------------------------------
# FACE covariance smoothing
# --------------------------------------------------------------------

def kl_curves(n=400, p=60, sigma=0.3, seed=0, lam=(1.0, 0.5)):
    t = np.linspace(0.0, 1.0, p)
    phi = np.column_stack([math.sqrt(2) * np.sin(2 * np.pi * t),
                           math.sqrt(2) * np.cos(2 * np.pi * t)])
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=(n, 2)) * np.sqrt(np.asarray(lam))
    return xi @ phi.T + rng.normal(scale=sigma, size=(n, p)), t, phi


def test_the_bspline_basis_is_a_partition_of_unity():
    B = bspline_basis(np.linspace(0, 1, 60), n_basis=12, degree=3)
    assert B.sum(axis=1) == pytest.approx(np.ones(60))
    assert np.all(B >= 0)
    assert B.shape == (60, 12)


def test_eigenvalues_recover_the_karhunen_loeve_truth():
    Y, t, _ = kl_curves()
    out = face_smooth(Y, t, n_basis=12)
    dt = float(t[1] - t[0])
    assert out["eigenvalues"][:2] * dt == pytest.approx([1.0, 0.5], abs=0.06)


def test_eigenfunctions_recover_the_truth():
    Y, t, phi = kl_curves()
    out = face_smooth(Y, t, n_basis=12)
    for k in range(2):
        c = abs(float(np.corrcoef(out["eigenfunctions"][:, k],
                                  phi[:, k])[0, 1]))
        assert c > 0.99


def test_the_noise_variance_is_estimated_not_absorbed():
    for sigma in (0.1, 0.3, 0.5):
        Y, t, _ = kl_curves(sigma=sigma)
        out = face_smooth(Y, t, n_basis=12)
        assert out["noise_variance"] == pytest.approx(sigma ** 2, rel=0.35)


def test_smoothing_through_the_diagonal_biases_it_upward():
    # the ridge of measurement error on the diagonal gets dragged into
    # the surface, which is why the diagonal is held out
    Y, t, _ = kl_curves(sigma=0.5)
    out = face_smooth(Y, t, n_basis=12)
    assert out["noise_variance_bias_if_kept"] > 0
    kept = np.diag(out["covariance_diagonal_kept"])
    held = np.diag(out["covariance"])
    assert float(np.mean(kept)) > float(np.mean(held))


def test_the_smoothed_covariance_is_essentially_positive_semidefinite():
    Y, t, _ = kl_curves()
    out = face_smooth(Y, t, n_basis=12)
    assert out["negative_eigenvalue_mass"] < 1e-2 * out["total_variance"]


def test_the_covariance_is_symmetric():
    Y, t, _ = kl_curves(n=100)
    C = face_smooth(Y, t, n_basis=10)["covariance"]
    assert C == pytest.approx(C.T)


def test_two_components_are_selected_for_a_two_component_process():
    Y, t, _ = kl_curves()
    assert face_smooth(Y, t, n_basis=12, pve=0.95)["npc"] == 2


def test_missing_observations_are_handled_pairwise():
    Y, t, _ = kl_curves(n=300)
    rng = np.random.default_rng(3)
    Y = Y.copy()
    Y[rng.random(Y.shape) < 0.2] = np.nan
    out = face_smooth(Y, t, n_basis=12)
    dt = float(t[1] - t[0])
    assert out["eigenvalues"][0] * dt == pytest.approx(1.0, abs=0.15)


def test_face_input_validation():
    Y, t, _ = kl_curves(n=20, p=20)
    with pytest.raises(ValueError, match="at least two curves"):
        face_smooth(Y[:1], t)
    with pytest.raises(ValueError, match="argvals has length"):
        face_smooth(Y, t[:5])
    with pytest.raises(ValueError, match="pve"):
        face_smooth(Y, t, pve=0.0)
    with pytest.raises(ValueError, match="n_basis must be at least"):
        bspline_basis(np.linspace(0, 1, 20), n_basis=2, degree=3)


# --------------------------------------------------------------------
# Identify, estimate, refute
# --------------------------------------------------------------------

def test_the_backdoor_criterion_on_a_confounder():
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True      # Z -> T, Z -> Y, T -> Y
    assert is_backdoor_admissible(A, 1, 2, []) is False
    assert is_backdoor_admissible(A, 1, 2, [0]) is True
    assert backdoor_sets(A, 1, 2) == [(0,)]


def test_a_mediator_is_not_admissible():
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[1, 2] = True                # T -> M -> Y
    assert is_backdoor_admissible(A, 0, 2, []) is True
    assert is_backdoor_admissible(A, 0, 2, [1]) is False


def test_conditioning_on_a_collider_opens_a_path():
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[2, 1] = True                # T -> C <- Y
    assert is_backdoor_admissible(A, 0, 2, []) is True
    assert backdoor_sets(A, 0, 2)[0] == ()


def test_the_effect_is_recovered_when_identified():
    rng = np.random.default_rng(0)
    n = 3000
    Z = rng.normal(size=n)
    T = 0.8 * Z + rng.normal(size=n)
    Y = 2.0 * T + 1.5 * Z + rng.normal(size=n)
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True
    out = model_identify_estimate_refute(A, np.column_stack([Z, T, Y]), 1, 2)
    assert out["identified"] is True
    assert out["adjustment_set"] == [0]
    assert out["estimate"] == pytest.approx(2.0, abs=0.1)


def test_the_unadjusted_estimate_is_biased_on_the_same_data():
    rng = np.random.default_rng(0)
    n = 3000
    Z = rng.normal(size=n)
    T = 0.8 * Z + rng.normal(size=n)
    Y = 2.0 * T + 1.5 * Z + rng.normal(size=n)
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True
    D = np.column_stack([Z, T, Y])
    forced = model_identify_estimate_refute(A, D, 1, 2, adjustment=[])
    assert abs(forced["estimate"] - 2.0) > 0.4
    assert forced["identified"] is False


def test_the_placebo_treatment_collapses_to_zero():
    rng = np.random.default_rng(1)
    n = 2000
    Z = rng.normal(size=n)
    T = 0.8 * Z + rng.normal(size=n)
    Y = 2.0 * T + 1.5 * Z + rng.normal(size=n)
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True
    out = model_identify_estimate_refute(A, np.column_stack([Z, T, Y]), 1, 2,
                                         n_refute=60)
    assert abs(out["placebo_effect"]) < 0.05
    assert out["passed_placebo"] is True


def test_a_random_common_cause_leaves_the_estimate_alone():
    rng = np.random.default_rng(2)
    n = 2000
    Z = rng.normal(size=n)
    T = 0.8 * Z + rng.normal(size=n)
    Y = 2.0 * T + 1.5 * Z + rng.normal(size=n)
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True
    out = model_identify_estimate_refute(A, np.column_stack([Z, T, Y]), 1, 2,
                                         n_refute=60)
    assert out["random_cause_effect"] == pytest.approx(out["estimate"],
                                                       abs=0.05)
    assert out["passed_random_cause"] is True


def test_conditioning_on_a_mediator_is_flagged_and_attenuates():
    rng = np.random.default_rng(3)
    n = 3000
    T = rng.normal(size=n)
    M = 1.0 * T + rng.normal(size=n)
    Y = 1.0 * M + rng.normal(size=n)       # total effect of T on Y is 1.0
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[1, 2] = True
    D = np.column_stack([T, M, Y])
    good = model_identify_estimate_refute(A, D, 0, 2, adjustment=[],
                                          n_refute=20)
    bad = model_identify_estimate_refute(A, D, 0, 2, adjustment=[1],
                                         n_refute=20)
    assert good["estimate"] == pytest.approx(1.0, abs=0.1)
    assert abs(bad["estimate"]) < 0.15          # the effect is absorbed
    assert bad["adjusted_for_mediator"] == [1]
    assert any("mediator" in w for w in bad.warnings)


def test_conditioning_on_a_collider_is_flagged():
    A = np.zeros((4, 4), bool)
    A[0, 2] = A[1, 2] = A[0, 3] = True      # T -> C <- U, T -> Y
    rng = np.random.default_rng(4)
    n = 2000
    T = rng.normal(size=n)
    U = rng.normal(size=n)
    C = T + U + rng.normal(size=n)
    Y = 1.0 * T + rng.normal(size=n)
    out = model_identify_estimate_refute(A, np.column_stack([T, U, C, Y]),
                                         0, 3, adjustment=[2], n_refute=20)
    assert out["adjusted_for_collider"] == [2]
    assert any("collider" in w for w in out.warnings)


def test_no_admissible_set_is_reported_as_unidentified():
    # unmeasured confounding: U is in the graph but its column is data
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True
    rng = np.random.default_rng(5)
    n = 500
    D = rng.normal(size=(n, 3))
    out = model_identify_estimate_refute(A, D, 1, 2, adjustment=[2],
                                         n_refute=10)
    assert out["identified"] is False
    assert any("back-door criterion is not satisfied" in w
               for w in out.warnings)


def test_dag_input_validation():
    A = np.zeros((3, 3), bool)
    A[0, 1] = A[0, 2] = A[1, 2] = True
    D = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(ValueError, match="must differ"):
        model_identify_estimate_refute(A, D, 1, 1)
    with pytest.raises(ValueError, match="self-loop"):
        B = A.copy()
        B[0, 0] = True
        model_identify_estimate_refute(B, D, 1, 2)
    with pytest.raises(ValueError, match="only estimator"):
        model_identify_estimate_refute(A, D, 1, 2, estimator="iv")


# --------------------------------------------------------------------
# W-NOMINATE
# --------------------------------------------------------------------

def chamber(n=120, m=220, dims=1, seed=3, beta=2.0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n, dims))
    zy = rng.normal(size=(m, dims)) * 0.8
    zn = rng.normal(size=(m, dims)) * 0.8
    eta = beta * (np.sum((truth[:, None, :] - zn[None, :, :]) ** 2, axis=2)
                  - np.sum((truth[:, None, :] - zy[None, :, :]) ** 2, axis=2))
    p = 0.5 * np.array([[math.erfc(-v / math.sqrt(2)) for v in row]
                        for row in eta])
    return (rng.random((n, m)) < p).astype(float), truth


def test_ideal_points_are_recovered_in_one_dimension():
    V, truth = chamber()
    out = wnominate_alternating(V, n_dims=1,
                                polarity=int(np.argmax(truth[:, 0])))
    r = abs(float(np.corrcoef(out["ideal_points"][:, 0], truth[:, 0])[0, 1]))
    assert r > 0.95
    assert out["converged"] is True


def test_the_svd_start_reaches_a_better_optimum_than_a_random_one():
    # the objective is not jointly concave, so the start matters: from
    # noise the fit sits at a materially worse likelihood
    V, truth = chamber()
    good = wnominate_alternating(V, n_dims=1, start="svd",
                                 polarity=int(np.argmax(truth[:, 0])))
    poor = wnominate_alternating(V, n_dims=1, start="random", seed=1,
                                 max_iter=60,
                                 polarity=int(np.argmax(truth[:, 0])))
    assert good["log_likelihood"] > poor["log_likelihood"]
    r_good = abs(float(np.corrcoef(good["ideal_points"][:, 0],
                                   truth[:, 0])[0, 1]))
    r_poor = abs(float(np.corrcoef(poor["ideal_points"][:, 0],
                                   truth[:, 0])[0, 1]))
    assert r_good > r_poor


def test_ideal_points_are_recovered_in_two_dimensions_up_to_rotation():
    V, truth = chamber(dims=2)
    out = wnominate_alternating(V, n_dims=2,
                                polarity=int(np.argmax(truth[:, 0])))
    A = truth - truth.mean(0)
    B = out["ideal_points"] - out["ideal_points"].mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    assert float(np.corrcoef((B @ R.T).ravel(), A.ravel())[0, 1]) > 0.95


def test_polarity_fixes_the_otherwise_arbitrary_sign():
    V, truth = chamber()
    k = int(np.argmax(truth[:, 0]))
    a = wnominate_alternating(V, n_dims=1, polarity=k)
    b = wnominate_alternating(V, n_dims=1, polarity=k, seed=99)
    assert a["ideal_points"][k, 0] > 0
    assert b["ideal_points"][k, 0] > 0
    assert float(np.corrcoef(a["ideal_points"][:, 0],
                             b["ideal_points"][:, 0])[0, 1]) > 0.95


def test_without_polarity_the_sign_is_reported_as_arbitrary():
    V, _ = chamber()
    out = wnominate_alternating(V, n_dims=1)
    assert any("identified only up to" in w for w in out.warnings)


def test_the_configuration_is_normalised():
    V, truth = chamber()
    out = wnominate_alternating(V, n_dims=1,
                                polarity=int(np.argmax(truth[:, 0])))
    x = out["ideal_points"]
    assert float(np.mean(x)) == pytest.approx(0.0, abs=1e-9)
    assert math.sqrt(float(np.mean(np.sum(x ** 2, axis=1)))) == \
        pytest.approx(1.0, abs=1e-9)


def test_unanimous_rollcalls_are_dropped():
    V, truth = chamber(m=100)
    V = np.column_stack([V, np.ones((V.shape[0], 5))])
    out = wnominate_alternating(V, n_dims=1,
                                polarity=int(np.argmax(truth[:, 0])))
    assert out["n_dropped_rollcalls"] == 5
    assert any("unanimous" in w for w in out.warnings)


def test_classification_beats_the_modal_baseline():
    V, truth = chamber()
    out = wnominate_alternating(V, n_dims=1,
                                polarity=int(np.argmax(truth[:, 0])))
    assert out["correct_classification"] > out["modal_baseline"]
    assert out["aggregate_pre"] > 0.3


def test_random_votes_give_no_reduction_in_error():
    rng = np.random.default_rng(9)
    V = (rng.random((80, 150)) < 0.5).astype(float)
    out = wnominate_alternating(V, n_dims=1, polarity=0)
    assert out["aggregate_pre"] < 0.35


def test_absences_are_tolerated():
    V, truth = chamber()
    rng = np.random.default_rng(11)
    V = V.copy()
    V[rng.random(V.shape) < 0.1] = np.nan
    out = wnominate_alternating(V, n_dims=1,
                                polarity=int(np.argmax(truth[:, 0])))
    r = abs(float(np.corrcoef(out["ideal_points"][:, 0], truth[:, 0])[0, 1]))
    assert r > 0.9


def test_voting_input_validation():
    V, _ = chamber(n=30, m=40)
    with pytest.raises(ValueError, match="must be 1 \\(yea\\)"):
        wnominate_alternating(V * 3, n_dims=1)
    with pytest.raises(ValueError, match="n_dims must be at least"):
        wnominate_alternating(V, n_dims=0)
    with pytest.raises(ValueError, match="polarity must lie"):
        wnominate_alternating(V, n_dims=1, polarity=999)
    with pytest.raises(ValueError, match='start must be'):
        wnominate_alternating(V, n_dims=1, start="pca")
    with pytest.raises(ValueError, match="roll calls divide"):
        wnominate_alternating(np.ones((10, 6)), n_dims=1)
