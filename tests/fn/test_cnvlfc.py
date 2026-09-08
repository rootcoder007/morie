"""Convergent cross mapping."""
import importlib

import pytest

C = importlib.import_module("morie.fn.cnvlfc")

UNI = C.coupled_logistic(400, bxy=0.0, byx=0.1)


def test_unidirectional_coupling_is_found_in_the_right_direction():
    r = C.ccm(UNI["x"], UNI["y"], E=2, exclude=1)
    assert r["verdict"] == "x drives y"


def test_the_skill_converges_with_library_length():
    r = C.ccm(UNI["x"], UNI["y"], E=2, exclude=1)
    fwd = [q["rho"] for q in r["x_causes_y"]["curve"]]
    assert r["x_causes_y"]["increase"] > 0.3
    assert all(fwd[i] <= fwd[i + 1] + 0.02
               for i in range(len(fwd) - 1))


def test_the_wrong_direction_does_not_converge():
    r = C.ccm(UNI["x"], UNI["y"], E=2, exclude=1)
    assert not r["y_causes_x"]["converges"]
    assert r["y_causes_x"]["rho_final"] < 0.3


def test_reversing_the_coupling_reverses_the_verdict():
    rev = C.coupled_logistic(400, bxy=0.1, byx=0.0)
    assert C.ccm(rev["x"], rev["y"], E=2, exclude=1)["verdict"] \
        == "y drives x"


def test_mutual_coupling_is_reported_as_mutual():
    bi = C.coupled_logistic(400, bxy=0.08, byx=0.08)
    r = C.ccm(bi["x"], bi["y"], E=2, exclude=1)
    assert r["x_causes_y"]["converges"]
    assert r["y_causes_x"]["converges"]


def test_uncoupled_systems_show_nothing():
    ind = C.coupled_logistic(400, rx=3.8, ry=3.65, bxy=0.0, byx=0.0,
                             x0=0.4, y0=0.7)
    r = C.ccm(ind["x"], ind["y"], E=2, exclude=1)
    assert r["verdict"].startswith("no convergent cross mapping")


def test_a_series_cross_maps_itself():
    assert C.cross_map(UNI["x"], UNI["x"], 2, 1,
                       exclude=1)["rho"] > 0.99


def test_a_longer_library_gives_more_skill():
    short = C.cross_map(UNI["x"], UNI["y"], 2, 1, library=20, seed=3,
                        exclude=1)["rho"]
    long_ = C.cross_map(UNI["x"], UNI["y"], 2, 1, exclude=1)["rho"]
    assert long_ > short + 0.1


def test_the_embedding_stacks_lags():
    em = C.embed([1.0, 2, 3, 4, 5, 6], E=3, tau=1)
    assert em["points"][0] == [3.0, 2.0, 1.0]
    assert em["index"][0] == 2
    assert len(em["points"]) == 4


def test_the_delay_is_honoured():
    em = C.embed(list(range(10)), E=2, tau=3)
    assert em["points"][0] == [3.0, 0.0]


def test_the_result_reports_its_settings():
    r = C.ccm(UNI["x"], UNI["y"], E=2, exclude=1)
    assert r["E"] == 2 and r["tau"] == 1
    assert len(r["lib_sizes"]) == len(r["x_causes_y"]["curve"])


@pytest.mark.parametrize("call", [
    lambda: C.embed([1.0, 2.0], E=3),
    lambda: C.embed([1.0] * 10, E=0),
    lambda: C.embed([1.0] * 10, tau=0),
    lambda: C.cross_map([1.0] * 10, [1.0] * 9),
    lambda: C.cross_map(UNI["x"], UNI["y"], 2, 1, library=2),
    lambda: C.cross_map(UNI["x"], UNI["y"], 2, 1, library=99999),
    lambda: C.coupled_logistic(50, rx=8.0),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
