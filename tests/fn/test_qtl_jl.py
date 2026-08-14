"""QTL mapping (interval, composite, genome scan) and JL projections."""
import importlib
import math

import pytest

J = importlib.import_module("morie.fn.qjlcrn")
Q = importlib.import_module("morie.fn.rqtmpl")
C = importlib.import_module("morie.fn.cqtmpl")
M = importlib.import_module("morie.fn.mqtmpl")
R = importlib.import_module("morie.fn.survrsf")


def backcross(n, seed, qtl_at=0.08, span=0.20, effect=2.0):
    rng = R._Rng(seed)
    left, right, y = [], [], []
    for _ in range(n):
        ml = 1 if rng.next() < 0.5 else 0
        q = (1 - ml) if rng.next() < Q.haldane(qtl_at) else ml
        mr = (1 - q) if rng.next() < Q.haldane(span - qtl_at) else q
        left.append(ml)
        right.append(mr)
        y.append(effect * q
                 + (rng.next() + rng.next() + rng.next() - 1.5) * 1.2)
    return left, right, y


LEFT, RIGHT, Y = backcross(200, 9)


# -------------------------------------------------------------- qjlcrn
def test_k0_is_the_theorem_formula():
    td = J.target_dimension(50, 0.3, 1.0)
    assert td["k0"] == pytest.approx(
        6.0 * math.log(50.0) / (0.09 / 2.0 - 0.027 / 3.0))
    assert td["failure_probability"] == pytest.approx(0.02)


@pytest.mark.parametrize("dist", J.DISTRIBUTIONS)
def test_both_coin_distributions_are_standardised(dist):
    m = J.moments(dist)
    assert m["mean"] == pytest.approx(0.0, abs=1e-15)
    assert m["variance"] == pytest.approx(1.0, abs=1e-14)


def test_the_sparse_distribution_touches_a_third():
    assert J.moments("sparse")["density"] == pytest.approx(1.0 / 3.0)
    assert J.moments("sparse")["fourth_moment"] == pytest.approx(3.0)


@pytest.mark.parametrize("bad", [
    lambda: J.target_dimension(1, 0.3),
    lambda: J.target_dimension(50, 0.0),
    lambda: J.target_dimension(50, 1.0),
    lambda: J.target_dimension(50, 0.3, 0.0),
    lambda: J.moments("gaussian"),
    lambda: J.projection_matrix(0, 5),
])
def test_invalid_projection_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        bad()


def test_the_projection_preserves_distances_at_the_bound():
    rng = R._Rng(6)
    A = [[rng.next() for _ in range(80)] for _ in range(20)]
    k = J.target_dimension(20, 0.4)["k"]
    e = J.project(A, k, "rademacher", seed=3)
    assert J.distortion(A, e["embedding"])["worst_distortion"] < 0.4


def test_a_tiny_k_breaks_the_guarantee():
    rng = R._Rng(6)
    A = [[rng.next() for _ in range(80)] for _ in range(20)]
    e = J.project(A, 3, "rademacher", seed=3)
    assert J.distortion(A, e["embedding"])["worst_distortion"] > 0.4


def test_ragged_input_is_refused():
    with pytest.raises(ValueError):
        J.project([[1.0, 2.0], [3.0]], 4)


# -------------------------------------------------------------- rqtmpl
def test_the_threshold_is_the_papers_value():
    assert Q.threshold(0.05)["threshold"] == pytest.approx(0.83,
                                                           abs=0.01)


def test_elod_exact_and_approximate():
    e = Q.elod(0.25, 1.0)
    assert e["elod"] == pytest.approx(0.5 * math.log10(1.25))
    assert e["approximation"] == pytest.approx(0.055)
    assert e["gap"] > 0.0


def test_progeny_required_is_the_ratio():
    p = Q.progeny_required(0.05, 1.0)
    assert p["n"] == pytest.approx(p["threshold"] / p["elod"])


def test_a_zero_effect_qtl_is_never_detected():
    with pytest.raises(ValueError):
        Q.progeny_required(0.0, 1.0)


def test_haldane_round_trip():
    assert Q.inverse_haldane(Q.haldane(0.3)) == pytest.approx(0.3)
    assert Q.haldane(0.0) == 0.0
    assert Q.haldane(10.0) < 0.5


@pytest.mark.parametrize("bad", [
    lambda: Q.haldane(-0.1),
    lambda: Q.inverse_haldane(0.5),
    lambda: Q.genotype_probabilities(0, 1, 0.6, 0.1),
    lambda: Q.threshold(0.0),
    lambda: Q.elod(1.0, 0.0),
])
def test_invalid_mapping_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        bad()


def test_genotype_probabilities_are_a_distribution():
    g = Q.genotype_probabilities(1, 0, 0.1, 0.2)
    assert sum(g) == pytest.approx(1.0)
    assert all(0.0 < v < 1.0 for v in g)


def test_at_a_marker_the_genotype_is_certain():
    assert Q.genotype_probabilities(1, 1, 0.0, 0.1)[1] == \
        pytest.approx(1.0)
    assert Q.genotype_probabilities(0, 1, 0.0, 0.1)[0] == \
        pytest.approx(1.0)


def test_the_two_lod_routes_agree():
    sm = Q.single_marker(Y, LEFT)
    assert sm["lod"] == pytest.approx(sm["lod_likelihood"], abs=1e-9)


def test_a_monomorphic_marker_is_refused():
    with pytest.raises(ValueError):
        Q.single_marker(Y, [1] * len(Y))


def test_interval_mapping_at_a_marker_is_single_marker():
    sm = Q.single_marker(Y, LEFT)
    im = Q.interval_map(Y, LEFT, RIGHT, 0.0, Q.haldane(0.20))
    assert im["lod"] == pytest.approx(sm["lod"], abs=1e-6)
    assert im["b"] == pytest.approx(sm["b"], abs=1e-6)


def test_the_scan_finds_the_planted_position():
    sc = Q.scan_interval(Y, LEFT, RIGHT, 0.20, step=0.02)
    assert abs(sc["peak_position"] - 0.08) < 0.03
    assert sc["peak_lod"] > Q.single_marker(Y, LEFT)["lod"]


def test_em_is_monotone():
    im = Q.interval_map(Y, LEFT, RIGHT, 0.05, 0.05)
    h = im["loglik_history"]
    assert all(h[i] <= h[i + 1] + 1e-9 for i in range(len(h) - 1))


# -------------------------------------------------------------- cqtmpl
def test_no_cofactors_is_interval_mapping():
    a = C.cim(Y, LEFT, RIGHT, 0.03, 0.05)
    b = Q.interval_map(Y, LEFT, RIGHT, 0.03, 0.05)
    assert a["lod"] == pytest.approx(b["lod"], abs=1e-9)


def test_a_wrong_length_cofactor_is_refused():
    with pytest.raises(ValueError):
        C.cim(Y, LEFT, RIGHT, 0.03, 0.05, [[1.0, 0.0]])


def test_cofactors_change_the_null_model():
    plain = C.cim(Y, LEFT, RIGHT, 0.03, 0.05)
    with_c = C.cim(Y, LEFT, RIGHT, 0.03, 0.05, [LEFT])
    assert with_c["sigma2_null"] < plain["sigma2_null"]
    assert with_c["n_cofactors"] == 1


def test_the_scan_needs_increasing_positions():
    with pytest.raises(ValueError):
        C.scan(Y, [LEFT, RIGHT], [0.2, 0.1])


def test_forward_selection_returns_distinct_markers():
    sel = C.select_cofactors(Y, [LEFT, RIGHT], k=2)
    assert len(set(sel["cofactors"])) == len(sel["cofactors"])


# -------------------------------------------------------------- mqtmpl
def test_the_hmm_posterior_is_a_distribution():
    hp = M.hmm_genotype_probabilities([[0, None, 1]],
                                      [0.0, 0.1, 0.2], 0.0)
    for p in hp[0]:
        assert sum(p) == pytest.approx(1.0)
    assert hp[0][0][0] == pytest.approx(1.0)


def test_a_missing_middle_marker_is_symmetric_between_flanks():
    hp = M.hmm_genotype_probabilities([[0, None, 1]],
                                      [0.0, 0.1, 0.2], 0.0)
    assert hp[0][1][0] == pytest.approx(0.5)


def test_an_error_rate_softens_every_call():
    hp = M.hmm_genotype_probabilities([[0, 0, 1]], [0.0, 0.1, 0.2],
                                      0.05)
    assert all(0.0 < p[1] < 1.0 for p in hp[0])


@pytest.mark.parametrize("bad", [
    lambda: M.hmm_genotype_probabilities([[0, 1]], [0.0, 0.1], 0.7),
    lambda: M.hmm_genotype_probabilities([[0, 1]], [0.1, 0.0]),
    lambda: M.hmm_genotype_probabilities([[0]], [0.0, 0.1]),
])
def test_invalid_hmm_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        bad()


def test_unsourced_scan_methods_are_refused():
    for m in ("hk", "imp"):
        assert not M.method_status(m)["available"]
        with pytest.raises(ValueError):
            M.scanone(Y, [LEFT, RIGHT], [0.0, 0.2], method=m)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        M.method_status("bayes")


def test_marker_regression_reports_at_markers_only():
    r = M.scanone(Y, [LEFT, RIGHT], [0.0, 0.2], method="mr")
    assert set(r["position"]) <= {0.0, 0.2}


def test_the_support_interval_brackets_the_peak():
    s = M.scanone(Y, [LEFT, RIGHT], [0.0, 0.2], method="em",
                  step=0.02)
    si = M.lod_support_interval(s, 1.5)
    assert si["lower"] <= si["peak"] <= si["upper"]


def test_the_permutation_threshold_is_a_quantile_of_its_own_nulls():
    p = M.permutation_threshold(Y, [LEFT, RIGHT], [0.0, 0.2],
                                n_perm=10, step=0.1, seed=1)
    assert p["threshold"] in p["null_maxima"]
    assert len(p["null_maxima"]) == 10


def test_an_invalid_alpha_is_refused():
    with pytest.raises(ValueError):
        M.permutation_threshold(Y, [LEFT, RIGHT], [0.0, 0.2],
                                n_perm=2, alpha=0.0)


# ------------------------------------------------- survrsf new rule
def test_log_rank_scores_are_savage_scores_without_censoring():
    t = [1.0, 2.0, 3.0, 4.0]
    s = R.logrank_scores(t, [1, 1, 1, 1])
    savage = [1.0 - sum(1.0 / (4 - j) for j in range(i + 1))
              for i in range(4)]
    assert s == pytest.approx(savage)
    assert sum(s) == pytest.approx(0.0, abs=1e-12)


def test_censored_observations_get_negative_scores():
    s = R.logrank_scores([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0])
    assert s[1] < 0.0 and s[3] < 0.0


def test_the_standardised_statistic_is_invariant_to_time_scale():
    t = [1.0, 2.0, 3.0, 4.0]
    e = [1, 1, 1, 1]
    a = R.logrank_score_statistic(t, e, [0, 0, 1, 1])
    b = R.logrank_score_statistic([10.0 * v for v in t], e,
                                  [0, 0, 1, 1])
    assert a == pytest.approx(b)


def test_an_empty_daughter_scores_zero():
    assert R.logrank_score_statistic([1.0, 2.0], [1, 1],
                                     [0, 0]) == 0.0


def test_the_score_rule_is_now_available_and_conserve_is_not():
    st = R.rule_status()
    assert "logrankscore" in st["available"]
    assert set(st["unavailable"]) == {"conserve"}
