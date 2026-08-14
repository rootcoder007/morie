"""randIE -- randomized interventional effects. Source: Didelez, Dawid
& Geneletti (2006), UAI 2006, 138-146, arXiv:1206.6840."""
import pytest

from morie.fn.randIE import (decompose, interventional_mean,
                             mediator_distribution,
                             randomized_interventional_effect)


def saturated():
    """A tiny, fully saturated table with no noise."""
    A, M, C, Y = [], [], [], []
    # P(M=1|A=1) = 0.5, P(M=1|A=0) = 0.0; E[Y|A,M] = 1 + 2A + 3M
    for a, m, reps in ((1, 1, 50), (1, 0, 50), (0, 0, 100)):
        for _ in range(reps):
            A.append(str(a))
            M.append(str(m))
            C.append("0")
            Y.append(1.0 + 2.0 * a + 3.0 * m)
    return Y, A, M, C


def test_mediator_distributions_sum_to_one():
    _, A, M, C = saturated()
    d = mediator_distribution(A, M, C)
    for cell in d["p"].values():
        assert sum(cell.values()) == pytest.approx(1.0, abs=1e-13)


def test_mediator_distribution_matches_the_planted_probability():
    _, A, M, C = saturated()
    d = mediator_distribution(A, M, C)
    assert d["p"][("1", "0")]["1"] == pytest.approx(0.5, abs=1e-13)
    assert d["p"][("0", "0")]["1"] == pytest.approx(0.0, abs=1e-13)


def test_total_equals_direct_plus_indirect_exactly():
    Y, A, M, C = saturated()
    r = randomized_interventional_effect(Y, A, M, C)
    assert decompose(r)["residual"] == pytest.approx(0.0, abs=1e-12)


def test_direct_effect_is_the_planted_treatment_coefficient():
    Y, A, M, C = saturated()
    r = randomized_interventional_effect(Y, A, M, C)
    assert r["direct"] == pytest.approx(2.0, abs=1e-12)


def test_indirect_effect_is_the_mediator_shift_times_its_coefficient():
    Y, A, M, C = saturated()
    r = randomized_interventional_effect(Y, A, M, C)
    assert r["indirect"] == pytest.approx(3.0 * 0.5, abs=1e-12)


def test_no_mediation_gives_a_zero_indirect_effect():
    # M is identical in both arms, so nothing flows through it.
    A = ["1"] * 100 + ["0"] * 100
    M = (["1"] * 50 + ["0"] * 50) * 2
    C = ["0"] * 200
    Y = [1.0 + 2.0 * int(A[i]) + 3.0 * int(M[i]) for i in range(200)]
    r = randomized_interventional_effect(Y, A, M, C)
    assert r["indirect"] == pytest.approx(0.0, abs=1e-12)


def test_both_routes_agree_on_a_saturated_table():
    Y, A, M, C = saturated()
    g = randomized_interventional_effect(Y, A, M, C, route="gformula")
    w = randomized_interventional_effect(Y, A, M, C, route="weighting")
    assert g["total"] == pytest.approx(w["total"], abs=1e-9)


def test_psi_with_matching_arms_is_reported_against_the_arm_mean():
    Y, A, M, C = saturated()
    p = interventional_mean(Y, A, M, C, a="1", a_star="1")
    # arm A=1 holds 50 units at Y = 1+2+3 = 6 and 50 at Y = 1+2+0 = 3
    assert p["own_mediator_mean"] == pytest.approx(4.5, abs=1e-12)
    assert p["estimate"] == pytest.approx(4.5, abs=1e-12)


def test_laplace_smoothing_changes_the_distribution():
    _, A, M, C = saturated()
    raw = mediator_distribution(A, M, C)["p"][("0", "0")]["1"]
    sm = mediator_distribution(A, M, C, laplace=1.0)["p"][("0", "0")]["1"]
    assert raw == 0.0 and sm > 0.0


def test_an_empty_gformula_cell_is_refused():
    # psi(0, 1) needs E[Y | A=0, M=1], and the control arm never takes
    # M=1 in this table, so the quantity is not identified.
    Y, A, M, C = saturated()
    with pytest.raises(ValueError):
        interventional_mean(Y, A, M, C, a="0", a_star="1")


def test_the_unidentified_control_arm_diagnostic_is_reported_as_none():
    Y, A, M, C = saturated()
    r = randomized_interventional_effect(Y, A, M, C)
    assert r["direct_control_arm"] is None
    assert r["total"] == pytest.approx(3.5, abs=1e-12)


def test_an_absent_arm_is_refused():
    Y, A, M, C = saturated()
    with pytest.raises(ValueError):
        interventional_mean(Y, A, M, C, a="9")


def test_an_unknown_route_is_refused():
    Y, A, M, C = saturated()
    with pytest.raises(ValueError):
        interventional_mean(Y, A, M, C, route="tmle")


def test_mismatched_lengths_are_refused():
    Y, A, M, C = saturated()
    with pytest.raises(ValueError):
        interventional_mean(Y, A[:-1], M, C)


def test_an_empty_mediator_vector_is_refused():
    with pytest.raises(ValueError):
        mediator_distribution([], [], [])
