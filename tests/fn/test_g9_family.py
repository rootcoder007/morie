"""Tests for sdpwts, mehtad and miprgr."""
import importlib
import math

import pytest

np = importlib.import_module("morie.fn._array_core")
sd = importlib.import_module("morie.fn.sdpwts")
mh = importlib.import_module("morie.fn.mehtad")
mp = importlib.import_module("morie.fn.miprgr")

SYM = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]]
F0 = [[1.0, 0.0], [0.0, 1.0]]
F1 = [[1.0, 0.0], [0.0, -1.0]]


# ---------------------------------------------------------------- sdpwts
def test_sdp_recovers_the_minimum_eigenvalue():
    r = sd.min_eigenvalue_sdp(SYM)
    assert r["error"] < 1e-5
    assert r["lambda_min"] == pytest.approx(min(np.linalg.eigh(SYM)[0]),
                                            abs=1e-12)


def test_sdp_lmi_assembly_and_psd_test():
    assert sd.lmi([0.5], F0, [F1]) == [[1.5, 0.0], [0.0, 0.5]]
    assert sd.is_psd([[1.0, 0.0], [0.0, 1.0]])["psd"]
    assert not sd.is_psd([[1.0, 2.0], [2.0, 1.0]])["psd"]
    with pytest.raises(ValueError):
        sd.lmi([0.5, 0.5], F0, [F1])


def test_sdp_barrier_is_infinite_outside_the_cone():
    assert not math.isfinite(sd.barrier([2.0], F0, [F1])["value"])
    assert sd.barrier([0.0], F0, [F1])["value"] == pytest.approx(0.0,
                                                                abs=1e-12)


def test_sdp_central_path_gap_is_m_over_t():
    assert sd.central_path_gap(100.0, 3)["gap"] == pytest.approx(0.03)
    with pytest.raises(ValueError):
        sd.central_path_gap(0.0, 3)


def test_sdp_requires_a_strictly_feasible_start():
    with pytest.raises(ValueError):
        sd.solve_sdp([1.0], F0, [F1], [2.0])
    with pytest.raises(ValueError):
        sd.solve_sdp([1.0], F0, [F1], [0.0], mu=1.0)


# ---------------------------------------------------------------- mehtad
AM = [[1.0, 1.0, 1.0, 0.0], [1.0, 3.0, 0.0, 1.0]]
BM = [4.0, 6.0]
CM = [-1.0, -2.0, 0.0, 0.0]


def test_mehtad_solves_the_lp_exactly():
    r = mh.solve_lp(AM, BM, CM)
    assert r["x"][0] == pytest.approx(3.0, abs=1e-6)
    assert r["x"][1] == pytest.approx(1.0, abs=1e-6)
    assert r["objective"] == pytest.approx(-5.0, abs=1e-6)
    assert r["converged"]


def test_mehtad_strong_duality_and_residuals():
    r = mh.solve_lp(AM, BM, CM)
    assert r["objective"] == pytest.approx(r["dual_objective"],
                                           abs=1e-6)
    assert r["primal_residual"] < 1e-8
    assert r["dual_residual"] < 1e-8


def test_mehtad_corrector_costs_no_more_iterations():
    a = mh.solve_lp(AM, BM, CM)
    b = mh.solve_lp(AM, BM, CM, corrector=False)
    assert a["objective"] == pytest.approx(b["objective"], abs=1e-6)
    assert a["iterations"] <= b["iterations"]


def test_mehtad_centering_parameter_is_the_cubed_ratio():
    assert mh.centering_parameter(1.0, 0.01)["sigma"] == \
        pytest.approx(1e-6)
    assert mh.centering_parameter(1.0, 0.9)["sigma"] == \
        pytest.approx(0.729)
    with pytest.raises(ValueError):
        mh.centering_parameter(1.0, 0.5, nu=20.0)
    with pytest.raises(ValueError):
        mh.centering_parameter(0.0, 0.5)


def test_mehtad_fraction_to_boundary():
    assert mh.max_step([1.0, 2.0], [-2.0, 1.0], 1.0) == \
        pytest.approx(0.5)
    step = mh.max_step([1.0, 2.0], [-2.0, 1.0])
    assert 1.0 - 2.0 * step > 0.0


def test_mehtad_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        mh.solve_lp(AM, [1.0], CM)


# ---------------------------------------------------------------- miprgr
AI = [[6.0, 4.0], [1.0, 2.0]]
BI = [24.0, 6.0]
CI = [5.0, 4.0]


def test_miprgr_relaxation_is_fractional_and_bounds():
    r = mp.solve_relaxation(AI, BI, CI, (), 2, True)
    assert r["value"] == pytest.approx(21.0, abs=1e-9)
    assert abs(r["x"][1] - round(r["x"][1])) > 0.1


def test_miprgr_matches_exhaustive_enumeration():
    bb = mp.branch_and_bound(AI, BI, CI, [0, 1])
    bf = mp.enumerate_integer(AI, BI, CI, [0, 1], upper=6)
    assert bb["value"] == pytest.approx(bf["value"], abs=1e-6)
    assert bb["x"] == bf["x"]


def test_miprgr_second_problem_matches_too():
    A2, b2, c2 = [[3.0, 2.0], [1.0, 4.0]], [12.0, 10.0], [4.0, 3.0]
    bb = mp.branch_and_bound(A2, b2, c2, [0, 1])
    bf = mp.enumerate_integer(A2, b2, c2, [0, 1], upper=6)
    assert bb["value"] == pytest.approx(bf["value"], abs=1e-6)


def test_miprgr_pruning_does_not_change_the_optimum():
    a = mp.branch_and_bound(AI, BI, CI, [0, 1])
    b = mp.branch_and_bound(AI, BI, CI, [0, 1], prune=False)
    assert a["value"] == pytest.approx(b["value"], abs=1e-6)
    assert a["nodes"] <= b["nodes"]


def test_miprgr_list_holds_the_path_not_the_frontier():
    bb = mp.branch_and_bound(AI, BI, CI, [0, 1])
    assert bb["max_list_length"] < bb["nodes"]


def test_miprgr_rounding_is_not_a_substitute():
    rel = mp.solve_relaxation(AI, BI, CI, (), 2, True)
    rnd = mp.round_relaxation(rel["x"], AI, BI, [0, 1])
    bb = mp.branch_and_bound(AI, BI, CI, [0, 1])
    val = sum(CI[j] * rnd["x"][j] for j in range(2))
    assert val < bb["value"] - 0.5


def test_miprgr_single_point_branch_is_feasible():
    r = mp.solve_relaxation(AI, BI, CI, ((0, "ge", 4.0),), 2)
    assert r["feasible"]
    assert r["value"] == pytest.approx(20.0, abs=1e-9)


def test_miprgr_rejects_bad_input():
    with pytest.raises(ValueError):
        mp.branch_and_bound(AI, BI, CI, [0, 5])
    with pytest.raises(ValueError):
        mp.solve_relaxation(AI, BI, CI, (), 2, True, solver="magic")
