"""Rangayyan pattern classification and CAD (bsaclass chunks 1 and 2).

The book's own Table 10.4 -- 18 abnormal VAG signals classified by direct
playback against a sonification method -- is used as the oracle for the
test of symmetry, and its stated 15/18 and 4/18 sensitivities are
reproduced from the same table.
"""

import math

import pytest

from fractions import Fraction

from morie.fn.bsaclass import (accuracy, bayescls, bayesnorm, bhattgauss,
                               bhattcoef, chernoff,
                               hellinger, kld,
                               divav, divergence, elbow, errbound,
                               fishcrit, fishlda, hclust, kfoldcv,
                               kmeans, knn, lindisc, lindsep, logreg,
                               loocv, mahal, mcnemar, normdist, ppv, qda,
                               roc, sens, sepindex, spec, svm, svmkern)

# Table 10.4: rows are direct playback, columns sonification, categories
# normal / indeterminate / abnormal, 18 truly abnormal VAG signals.
TABLE_10_4 = [[2, 0, 10], [1, 0, 1], [0, 0, 4]]

X2 = [[1, 1], [1.4, 0.8], [0.7, 1.3], [1.1, 1.6],
      [3, 3], [3.3, 2.7], [2.6, 3.2], [3.1, 3.5]]
Y2 = [0, 0, 0, 0, 1, 1, 1, 1]
C1 = [[2, 0.3], [0.3, 1]]
C2 = [[1, -0.2], [-0.2, 3]]


# ------------------------------------------------- the accuracy measures

def test_sensitivity_eq10100_ignores_false_alarms():
    r = sens(45, 5)
    assert r["sensitivity"] == pytest.approx(0.9)
    assert r["fnf"] == pytest.approx(0.1)
    # a test calling everyone positive scores a perfect sensitivity
    assert sens(50, 0)["sensitivity"] == 1.0
    assert r["says_nothing_about_false_alarms"] is True


def test_specificity_eq10101_is_the_mirror():
    r = spec(40, 10)
    assert r["specificity"] == pytest.approx(0.8)
    assert r["fpf"] == pytest.approx(0.2)
    # the book's identities
    assert r["specificity"] + r["fpf"] == pytest.approx(1.0)
    assert sens(45, 5)["sensitivity"] + sens(45, 5)["fnf"] == \
        pytest.approx(1.0)


def test_sensitivity_and_specificity_accept_a_table():
    t = [[45, 5], [10, 40]]
    assert sens(t)["sensitivity"] == pytest.approx(0.9)
    assert spec(t)["specificity"] == pytest.approx(0.8)


def test_ppv_eq10106_and_its_dependence_on_prevalence():
    r = ppv(45, 10)
    assert r["ppv"] == pytest.approx(45 / 55)
    assert r["depends_on_prevalence"] is True
    # the same test on a rare disease has a far worse PPV
    rare = ppv(45, 10, prevalence=0.001, sensitivity=0.9,
               specificity=0.8)["ppv_at_prevalence"]
    assert rare < 0.01


def test_accuracy_eq10102_is_prevalence_weighted():
    # the book gives the weighted form first and the raw one as a fallback
    r = accuracy(tp=45, tn=40, fp=10, fn=5)
    assert r["raw_accuracy"] == pytest.approx(0.85)
    assert r["prior_weighted"] is False
    w = accuracy(tp=45, tn=40, fp=10, fn=5, prevalence=0.01)
    assert w["accuracy"] == pytest.approx(0.9 * 0.01 + 0.8 * 0.99)
    assert w["prior_weighted"] is True
    assert w["accuracy"] != pytest.approx(r["raw_accuracy"])


def test_eq10103_is_eq10102_at_the_test_set_prevalence():
    r = accuracy(tp=45, tn=40, fp=10, fn=5)
    at_test = accuracy(tp=45, tn=40, fp=10, fn=5,
                       prevalence=r["test_set_prevalence"])
    assert at_test["accuracy"] == pytest.approx(r["raw_accuracy"])
    assert r["eq_10_103_is_eq_10_102_at_the_test_set_prevalence"] is True


def test_accuracy_refuses_an_empty_class_or_table():
    with pytest.raises(ValueError):
        accuracy(tp=0, tn=5, fp=0, fn=0)
    with pytest.raises(ValueError):
        accuracy(tp=0, tn=0, fp=0, fn=0)
    with pytest.raises(ValueError):
        accuracy(table=[[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------- the ROC

def test_roc_area_equals_the_mann_whitney_statistic():
    s = [0.9, 0.75, 0.7, 0.62, 0.55, 0.4, 0.35, 0.2]
    lab = [1, 1, 0, 1, 0, 1, 0, 0]
    r = roc(s, lab)
    assert r["trapezoidal_equals_mann_whitney"] is True
    assert r["auc"] == pytest.approx(r["mann_whitney"])
    assert 0.0 <= r["auc"] <= 1.0


def test_a_perfect_and_a_useless_test():
    perfect = roc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert perfect["auc"] == pytest.approx(1.0)
    tied = roc([0.5] * 4, [1, 1, 0, 0])
    assert tied["auc"] == pytest.approx(0.5)
    assert tied["ties_counted_as_half"] is True


def test_the_roc_needs_both_classes():
    with pytest.raises(ValueError):
        roc([0.1, 0.2], [1, 1])


# ----------------------------------------------- the test of symmetry

def test_mcnemar_reproduces_the_books_table_10_4():
    r = mcnemar(TABLE_10_4)
    assert r["statistic"] == pytest.approx(12.0)
    assert r["df"] == 3
    assert r["p_value"] == pytest.approx(0.007383, abs=1e-6)
    assert r["is_bowker"] is True
    assert r["n"] == 18


def test_the_books_stated_sensitivities_follow_from_the_same_table():
    # the book: sonification 15/18 abnormal, direct playback 4/18
    n = sum(sum(row) for row in TABLE_10_4)
    assert sum(row[2] for row in TABLE_10_4) == 15
    assert sum(TABLE_10_4[2]) == 4
    assert n == 18


def test_mcnemar_2x2_uses_the_continuity_correction():
    r = mcnemar([[20, 5], [8, 25]])
    assert r["df"] == 1
    assert r["is_bowker"] is False
    assert r["continuity_correction"] is True
    # (|5-8|-1)^2 / 13
    assert r["statistic"] == pytest.approx(4.0 / 13.0)
    off = mcnemar([[20, 5], [8, 25]], correct=False)
    assert off["statistic"] == pytest.approx(9.0 / 13.0)


def test_only_the_off_diagonal_matters():
    a = mcnemar([[2, 0, 10], [1, 0, 1], [0, 0, 4]])
    b = mcnemar([[99, 0, 10], [1, 99, 1], [0, 0, 99]])
    assert a["statistic"] == pytest.approx(b["statistic"])
    assert a["diagonal_contributes_nothing"] is True


def test_a_symmetric_table_is_refused_and_a_ragged_one_too():
    with pytest.raises(ValueError):
        mcnemar([[5, 0], [0, 5]])
    with pytest.raises(ValueError):
        mcnemar([[1, 2, 3], [4, 5, 6]])


# ----------------------------------------------- separability of features

def test_normdist_eq10112_divides_by_the_sum_of_the_sds():
    r = normdist(0.0, 2.0, 1.0, 1.0)
    assert r["dn"] == pytest.approx(1.0)          # 2 / (1 + 1)
    assert r["denominator_is_the_sum_not_the_quadrature_sum"] is True


def test_normdist_has_the_blind_spot_the_book_states():
    r = normdist(1.0, 1.0, 1.0, 5.0)
    assert r["dn"] == 0.0
    assert r["blind_to_variance_when_means_match"] is True


def test_divergence_eq10117_does_not_have_that_blind_spot():
    d = divergence([0, 0], [0, 0], [[1, 0], [0, 1]], [[3, 0], [0, 3]])
    assert d["divergence"] > 0
    assert d["mean_term"] == pytest.approx(0.0)
    assert d["separates_equal_means_via_the_covariance_term"] is True


def test_divergence_is_symmetric_and_vanishes_for_identical_pdfs():
    a = divergence([0, 1], [2, -1], C1, C2)["divergence"]
    b = divergence([2, -1], [0, 1], C2, C1)["divergence"]
    assert a == pytest.approx(b)
    same = divergence([1, 2], [1, 2], C1, C1)
    assert same["divergence"] == pytest.approx(0.0, abs=1e-12)
    assert same["zero_for_identical_pdfs"] is True


def test_divergence_is_additive_over_independent_features():
    # two independent features: the 2-D divergence is the sum of the 1-D
    d1 = divergence([0], [1], [[1]], [[2]])["divergence"]
    d2 = divergence([0], [3], [[1]], [[1]])["divergence"]
    joint = divergence([0, 0], [1, 3], [[1, 0], [0, 1]],
                       [[2, 0], [0, 1]])["divergence"]
    assert joint == pytest.approx(d1 + d2)


def test_divav_reports_the_worst_pair_not_only_the_average():
    r = divav([[0, 0], [0.05, 0], [8, 8]],
              [[[1, 0], [0, 1]]] * 3)
    assert r["minimum"] < r["average"]
    assert r["worst_pair"] == (0, 1)
    assert r["average_hides_the_worst_pair"] is True
    with pytest.raises(ValueError):
        divav([[0, 0]], [[[1, 0], [0, 1]]])


def test_bhattacharyya_is_documented_as_not_from_this_book():
    r = bhattgauss([0, 1], [2, -1], C1, C2)
    assert r["not_from_this_book"] is True
    assert r["book_uses_divergence_eq_10_115"] is True
    assert r["bhattacharyya"] > 0
    # identical distributions give zero
    assert bhattgauss([1, 2], [1, 2], C1, C1)["bhattacharyya"] == \
        pytest.approx(0.0, abs=1e-12)


def test_the_error_bound_pairs_with_bhattacharyya():
    r = errbound(0.5, 0.5, 1.4)
    assert r["bound"] == pytest.approx(0.5 * math.exp(-1.4))
    assert r["not_from_this_book"] is True
    assert r["pairs_with_bhatt_not_with_divergence"] is True
    assert r["tightest_at_equal_priors"] is True
    with pytest.raises(ValueError):
        errbound(0.3, 0.3, 1.0)          # priors do not sum to 1


def test_fisher_criterion_is_not_the_books_normalized_distance():
    r = fishcrit([1, 2, 3, 4], [4, 5, 6, 8])
    assert r["is_not_eq_10_112"] is True
    # they agree in ranking only for equal spread
    equal = fishcrit([1, 2, 3, 4], [5, 6, 7, 8])
    assert equal["agrees_with_eq_10_112_ranking_only_for_equal_spread"] \
        is True


def test_separability_index_grows_as_the_classes_separate():
    near = sepindex([[0, 0], [0.1, 0], [1, 0], [1.1, 0]], [0, 0, 1, 1])
    far = sepindex([[0, 0], [0.1, 0], [9, 0], [9.1, 0]], [0, 0, 1, 1])
    assert far["j"] > near["j"]
    assert near["ignores_off_diagonal_structure"] is True
    with pytest.raises(ValueError):
        sepindex([[0, 0], [0, 0]], [0, 0])        # one class


# ------------------------------------------------------ the discriminants

def test_fisher_lda_separates_and_is_two_class_only():
    r = fishlda(X2, Y2)
    assert r["classes"] == [0, 1]
    assert r["two_class_only"] is True
    assert r["criterion"] > 0
    with pytest.raises(ValueError):
        fishlda(X2 + [[9, 9], [9.1, 9]], Y2 + [2, 2])


def test_lindsep_fits_a_threshold_at_least_as_good_as_the_midpoint():
    r = lindsep(X2, Y2)
    assert r["training_errors"] <= r["midpoint_errors"]
    assert r["resubstitution_error_is_optimistic"] is True
    assert 0.0 <= r["training_accuracy"] <= 1.0


def test_lindisc_assigns_to_the_largest_discriminant():
    r = lindisc([1, 2], [[1, 0], [0, 1]], [0.1, -0.1])
    assert r["d"] == pytest.approx([1.1, 1.9])
    assert r["assigned"] == 1
    assert r["margin"] == pytest.approx(0.8)
    assert r["regions_are_convex"] is True


def test_mahalanobis_differs_from_euclidean_under_correlation():
    r = mahal([2, 0], [0, 0], [[4, 0], [0, 1]])
    assert r["distance"] == pytest.approx(1.0)     # 2 / sqrt(4)
    assert r["euclidean"] == pytest.approx(2.0)
    assert r["differs_from_euclidean"] is True


def test_mahalanobis_reduces_to_euclidean_for_the_identity():
    r = mahal([3, 4], [0, 0], [[1, 0], [0, 1]])
    assert r["distance"] == pytest.approx(5.0)
    assert r["euclidean"] == pytest.approx(5.0)


# -------------------------------------------------------- the classifiers

def test_knn_eq10029_and_the_outlier_warning():
    assert knn(X2, Y2, [1.1, 1.0])["assigned"] == 0
    assert knn(X2, Y2, [3.1, 3.0])["assigned"] == 1
    assert knn(X2, Y2, [1.1, 1.0])["single_neighbour_may_be_an_outlier"] \
        is True
    assert knn(X2, Y2, [1.1, 1.0], k=3)["k"] == 3


def test_knn_reports_a_tied_vote_rather_than_hiding_it():
    r = knn(X2, Y2, [2.05, 2.05], k=2)
    assert r["tie"] is True
    assert len(r["tied_classes"]) == 2


def test_knn_mahalanobis_needs_the_covariance():
    with pytest.raises(ValueError):
        knn(X2, Y2, [1, 1], metric="mahalanobis")
    r = knn(X2, Y2, [1.1, 1.0], metric="mahalanobis",
            C=[[1, 0], [0, 1]])
    assert r["assigned"] == 0
    with pytest.raises(ValueError):
        knn(X2, Y2, [1, 1], k=99)


def test_bayes_prior_can_overturn_the_likelihood():
    r = bayescls([0.6, 0.4], [0.1, 0.9])
    assert r["maximum_likelihood_choice"] == 0
    assert r["assigned"] == 1
    assert r["prior_changed_the_decision"] is True
    assert sum(r["posterior"]) == pytest.approx(1.0)


def test_bayes_normal_log_form_and_the_dropped_constant():
    r = bayesnorm([0.5, 0.5], [[0, 0], [3, 3]], [C1, C2])
    n = 2
    const = 0.5 * n * math.log(2 * math.pi)
    assert r["constant_term"] == pytest.approx(const)
    for a, b in zip(r["d_full"], r["d_dropped_constant"]):
        assert a == pytest.approx(b - const)
    # dropping it cannot change the ranking
    assert (max(range(2), key=lambda i: r["d_full"][i])
            == max(range(2), key=lambda i: r["d_dropped_constant"][i]))


def test_equal_covariances_make_the_boundary_linear():
    same = bayesnorm([0.5, 0.5], [[0, 0], [3, 3]],
                     [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    assert same["linear_when_covariances_are_equal"] is True
    diff = bayesnorm([0.5, 0.5], [[0, 0], [3, 3]], [C1, C2])
    assert diff["linear_when_covariances_are_equal"] is False
    assert diff["surfaces_are_hyperquadrics"] is True


def test_qda_classifies_and_counts_its_own_parameters():
    assert qda(X2, Y2, [1.1, 1.0])["assigned"] == 0
    assert qda(X2, Y2, [3.1, 3.0])["assigned"] == 1
    assert qda(X2, Y2, [1, 1])["parameters_per_class"] == 3   # p(p+1)/2


def test_qda_refuses_a_class_with_too_few_samples():
    with pytest.raises(ValueError) as e:
        qda([[0, 0], [1, 1], [5, 5], [6, 6], [7, 8]],
            [0, 0, 1, 1, 1], [1, 1])
    assert "more samples than features" in str(e.value)


def test_logistic_regression_separates_and_reports_the_fit():
    r = logreg(X2, Y2)
    assert r["training_accuracy"] == pytest.approx(1.0)
    assert r["loglik"] < 0
    assert r["models_the_posterior_directly"] is True
    assert len(r["coefficients"]) == 2
    with pytest.raises(ValueError):
        logreg(X2, [0, 0, 0, 0, 2, 2, 2, 2])       # not 0/1


def test_kmeans_finds_the_two_groups_and_is_reproducible():
    a = kmeans(X2, 2)
    b = kmeans(X2, 2)
    assert a["labels"] == b["labels"]
    assert sorted(a["sizes"]) == [4, 4]
    assert a["local_minimum_only"] is True
    assert a["depends_on_the_starting_centroids"] is True
    with pytest.raises(ValueError):
        kmeans(X2, 99)


def test_kmeans_wcss_falls_as_k_rises():
    v = [kmeans(X2, k)["wcss"] for k in (1, 2, 3, 4)]
    assert all(b <= a + 1e-9 for a, b in zip(v, v[1:]))


def test_the_elbow_finds_two_clusters():
    r = elbow(X2, kmax=5)
    assert r["knee"] == 2
    assert r["monotonic"] is True
    assert r["wcss_cannot_be_minimized"] is True
    assert r["heuristic_only"] is True


def test_hierarchical_linkage_changes_the_partition():
    a = hclust(X2, "single", k=3)["labels"]
    b = hclust(X2, "complete", k=3)["labels"]
    assert a != b
    assert hclust(X2, "single")["single_linkage_chains"] is True
    # at k = 2 all three linkages recover the obvious split
    for link in ("single", "complete", "average"):
        lab = hclust(X2, link, k=2)["labels"]
        assert lab[:4] == [lab[0]] * 4
        assert lab[4:] == [lab[4]] * 4
        assert lab[0] != lab[4]


def test_hierarchical_merges_are_monotone_for_these_linkages():
    for link in ("single", "complete", "average"):
        assert hclust(X2, link)["monotonic_merges"] is True
    with pytest.raises(ValueError):
        hclust(X2, "ward")


def test_cross_validation_is_honest_about_separate_data():
    r = kfoldcv(X2, Y2, k=4)
    assert r["accuracy"] == pytest.approx(1.0)
    assert r["stratified"] is True
    assert r["train_and_test_must_be_separate"] is True
    assert sum(f["n"] for f in r["per_fold"]) == len(X2)


def test_leave_one_out_is_deterministic():
    a = loocv(X2, Y2)
    b = loocv(X2, Y2)
    assert a["error_rate"] == b["error_rate"]
    assert a["n_fits"] == len(X2)
    assert a["deterministic"] is True
    assert a["high_variance"] is True


def test_cross_validation_catches_a_classifier_that_only_memorizes():
    # random labels: resubstitution with 1-NN is perfect, LOO is not
    X = [[i, 0] for i in range(10)]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert loocv(X, y)["accuracy"] < 0.6


def test_svm_uses_only_the_support_vectors():
    r = svm([[1, 1], [2, 2], [4, 4], [5, 5]], [-1, -1, 1, 1])
    assert r["training_accuracy"] == pytest.approx(1.0)
    assert r["n_support"] >= 2
    assert r["margin"] > 0
    assert r["boundary_set_by_the_support_vectors_only"] is True
    with pytest.raises(ValueError):
        svm([[1, 1], [2, 2]], [0, 1])              # needs -1/+1


def test_the_kernel_trick_solves_xor_where_the_linear_svm_cannot():
    xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    yx = [-1, 1, 1, -1]
    rbf = svmkern(xor, yx, kernel="rbf", gamma=1.0, C=10.0)
    assert rbf["training_accuracy"] == pytest.approx(1.0)
    assert svm(xor, yx, C=10.0)["training_accuracy"] < 1.0
    assert rbf["no_weight_vector_in_the_original_space"] is True
    assert rbf["model_grows_with_the_training_set"] is True


def test_the_kernel_svm_classifies_a_query_and_flags_the_sigmoid():
    xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    yx = [-1, 1, 1, -1]
    r = svmkern(xor, yx, query=[0, 0], kernel="rbf", gamma=1.0, C=10.0)
    assert r["assigned"] == -1.0
    sig = svmkern(xor, yx, kernel="sigmoid", gamma=0.5, C=1.0)
    assert sig["sigmoid_kernel_is_not_always_positive_definite"] is True
    with pytest.raises(ValueError):
        svmkern(xor, yx, kernel="cosine")


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsaclass import (rangayyan_accuracy,
                                   rangayyan_bhattacharyya,
                                   rangayyan_knn_classifier,
                                   rangayyan_mcnemar_test)
    assert rangayyan_accuracy(tp=45, tn=40, fp=10, fn=5)["raw_accuracy"] \
        == pytest.approx(0.85)
    assert rangayyan_mcnemar_test(TABLE_10_4)["statistic"] == \
        pytest.approx(12.0)
    assert rangayyan_knn_classifier(X2, Y2, [1.1, 1.0])["assigned"] == 0
    # the pre-policy Bhattacharyya name now reaches the book's divergence
    assert "divergence" in rangayyan_bhattacharyya(
        [0, 1], [2, -1], C1, C2)


# ------------------------------- the accuracy forms and exact arithmetic

def test_accuracy_offers_every_definition_at_once():
    r = accuracy(tp=45, tn=40, fp=10, fn=5, prevalence=0.01)
    assert r["raw_accuracy"] == pytest.approx(0.85)
    assert r["balanced_accuracy"] == pytest.approx(0.85)
    assert r["weighted_accuracy"] == pytest.approx(0.801)
    # all three are always present, whichever is the headline
    assert r["kind"] == "weighted"
    assert r["accuracy"] == r["weighted_accuracy"]


def test_the_headline_form_is_selectable():
    for kind in ("raw", "balanced"):
        r = accuracy(tp=45, tn=40, fp=10, fn=5, kind=kind)
        assert r["kind"] == kind
        assert r["accuracy"] == r[kind + "_accuracy"]
    # weighted without a prevalence is refused rather than guessed
    with pytest.raises(ValueError):
        accuracy(tp=45, tn=40, fp=10, fn=5, kind="weighted")
    with pytest.raises(ValueError):
        accuracy(tp=45, tn=40, fp=10, fn=5, kind="f1")


def test_balanced_accuracy_is_eq10102_at_one_half():
    a = accuracy(tp=45, tn=40, fp=10, fn=5, exact=True)
    b = accuracy(tp=45, tn=40, fp=10, fn=5, prevalence="0.5", exact=True)
    assert a["balanced_accuracy"] == b["weighted_accuracy"]
    assert a["balanced_is_eq_10_102_at_one_half"] is True


def test_exact_accuracy_is_a_rational_not_a_float():
    r = accuracy(tp=45, tn=40, fp=10, fn=5, exact=True)
    assert isinstance(r["raw_accuracy"], Fraction)
    assert r["raw_accuracy"] == Fraction(17, 20)
    assert float(r["raw_accuracy"]) == pytest.approx(0.85)
    w = accuracy(tp=45, tn=40, fp=10, fn=5, prevalence="0.01", exact=True)
    assert w["accuracy"] == Fraction(801, 1000)


def test_exact_arithmetic_avoids_a_representation_error():
    # 1/3 has no exact float; the rational form does
    r = accuracy(tp=1, tn=1, fp=1, fn=2, exact=True)
    # one third has no finite binary representation, so the float is a
    # nearby value and the rational is the number itself
    assert r["sensitivity"] == Fraction(1, 3)
    approx = accuracy(tp=1, tn=1, fp=1, fn=2)["sensitivity"]
    assert Fraction(approx) != Fraction(1, 3)
    assert abs(Fraction(approx) - Fraction(1, 3)) < Fraction(1, 10 ** 15)


def test_accuracy_refuses_fractional_counts():
    with pytest.raises(ValueError):
        accuracy(tp=45.5, tn=40, fp=10, fn=5)


# ------------------------------------ the book's KLD and the coefficient

def test_kld_eq533_is_weighted_by_the_second_pdf():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    r = kld(p1, p2)
    manual = sum(p2[i] * math.log(p2[i] / p1[i]) for i in range(3))
    assert r["kld"] == pytest.approx(manual)
    assert r["weighted_by_the_second_pdf"] is True


def test_kld_is_asymmetric_and_its_sum_is_the_divergence():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    r = kld(p1, p2)
    assert r["asymmetric"] is True
    assert r["kld"] != pytest.approx(r["reversed"])
    assert r["symmetric_sum"] == pytest.approx(r["kld"] + r["reversed"])
    assert r["symmetric_sum_is_the_divergence_of_eq_10_115"] is True


def test_kld_vanishes_for_identical_pdfs_and_is_nonnegative():
    p = [0.25, 0.25, 0.5]
    assert kld(p, p)["kld"] == pytest.approx(0.0, abs=1e-15)
    assert kld([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])["kld"] > 0


def test_kld_refuses_a_zero_where_the_weighting_pdf_is_positive():
    with pytest.raises(ValueError):
        kld([0.0, 0.5, 0.5], [0.3, 0.3, 0.4])
    with pytest.raises(ValueError):
        kld([0.5, 0.5], [0.3, 0.3, 0.4])


def test_the_bhattacharyya_coefficient_is_an_overlap_in_the_unit_interval():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    r = bhattcoef(p1, p2)
    assert 0.0 <= r["coefficient"] <= 1.0
    assert r["in_unit_interval"] is True
    assert bhattcoef(p1, p1)["coefficient"] == pytest.approx(1.0)
    assert bhattcoef(p1, p1)["identical"] is True
    disjoint = bhattcoef([1.0, 0.0], [0.0, 1.0])
    assert disjoint["coefficient"] == pytest.approx(0.0)
    assert disjoint["disjoint"] is True


def test_the_distance_is_minus_the_log_of_the_coefficient():
    r = bhattcoef([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])
    assert r["distance"] == pytest.approx(-math.log(r["coefficient"]))
    assert r["the_overlap_is_where_errors_must_happen"] is True
    assert r["not_from_this_book"] is True
    assert bhattcoef([1.0, 0.0], [0.0, 1.0])["distance"] == float("inf")


def test_the_error_bound_tightens_as_the_overlap_falls():
    close = bhattcoef([0.5, 0.5], [0.45, 0.55])["distance"]
    far = bhattcoef([0.9, 0.1], [0.1, 0.9])["distance"]
    assert errbound(0.5, 0.5, far)["bound"] < \
        errbound(0.5, 0.5, close)["bound"]


# ------------------------------- the rest of the divergence family

def test_chernoff_at_one_half_is_the_bhattacharyya_coefficient():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    bc = bhattcoef(p1, p2)["coefficient"]
    at_half = chernoff(p1, p2, alpha=0.5)
    assert at_half["coefficient"] == pytest.approx(bc, abs=1e-12)
    assert at_half["bhattacharyya_is_alpha_one_half"] is True
    assert at_half["alpha_searched"] is False


def test_the_searched_chernoff_is_at_least_as_tight():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    best = chernoff(p1, p2)
    assert best["coefficient"] <= bhattcoef(p1, p2)["coefficient"] + 1e-12
    assert best["at_least_as_tight_as_bhattacharyya"] is True
    assert best["alpha_searched"] is True
    assert 0.0 <= best["alpha"] <= 1.0
    with pytest.raises(ValueError):
        chernoff(p1, p2, alpha=1.5)


def test_hellinger_squared_is_one_minus_the_coefficient():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    h = hellinger(p1, p2)
    assert h["squared"] == pytest.approx(
        1.0 - bhattcoef(p1, p2)["coefficient"], abs=1e-12)
    assert h["identity_h2_equals_one_minus_bc"] is True
    assert 0.0 <= h["hellinger"] <= 1.0
    assert hellinger(p1, p1)["hellinger"] == pytest.approx(0.0, abs=1e-12)
    assert hellinger([1.0, 0.0], [0.0, 1.0])["hellinger"] == \
        pytest.approx(1.0)


def test_hellinger_is_a_metric_where_the_bhattacharyya_distance_is_not():
    a = [0.2, 0.3, 0.5]
    b = [0.1, 0.4, 0.5]
    c = [0.5, 0.25, 0.25]
    hab = hellinger(a, b)["hellinger"]
    hbc = hellinger(b, c)["hellinger"]
    hac = hellinger(a, c)["hellinger"]
    assert hac <= hab + hbc + 1e-12          # the triangle inequality
    assert hellinger(a, b)["hellinger"] == pytest.approx(
        hellinger(b, a)["hellinger"])        # symmetric
    assert hellinger(a, b)["is_a_true_metric"] is True
    assert hellinger(a, b)["bhattacharyya_distance_does_not"] is True


def test_every_borrowed_measure_carries_its_primary_citation():
    p1 = [0.2, 0.3, 0.5]
    p2 = [0.1, 0.4, 0.5]
    for r in (bhattcoef(p1, p2), chernoff(p1, p2), hellinger(p1, p2),
              errbound(0.5, 0.5, 1.0),
              bhattgauss([0, 1], [2, -1], C1, C2)):
        assert r["not_from_this_book"] is True
        assert "reference" in r
        assert len(r["reference"]) > 40
    assert "1943" in bhattcoef(p1, p2)["reference"]
    assert "493-507" in chernoff(p1, p2)["reference"]
    assert "1909" in hellinger(p1, p2)["reference"]
    assert "1967" in errbound(0.5, 0.5, 1.0)["reference"]
