"""Free-Wilson additive substituent analysis."""
import importlib

import pytest

F = importlib.import_module("morie.fn.frwil")

TRUE = {"mu": 5.0, "Cl": 0.8, "Br": 1.1, "Me": 0.5, "Et": 0.9}
P1 = ["H", "Cl", "Br"]
P2 = ["H", "Me", "Et"]
C, Y = [], []
for _g1 in P1:
    for _g2 in P2:
        C.append((_g1, _g2))
        Y.append(TRUE["mu"] + TRUE.get(_g1, 0.0) + TRUE.get(_g2, 0.0))


def test_the_reference_fit_recovers_the_true_contributions():
    f = F.free_wilson(C, Y)
    assert f["coefficients"]["intercept"] == pytest.approx(5.0)
    assert f["coefficients"]["P1:Cl"] == pytest.approx(0.8)
    assert f["coefficients"]["P1:Br"] == pytest.approx(1.1)
    assert f["coefficients"]["P2:Me"] == pytest.approx(0.5)
    assert f["coefficients"]["P2:Et"] == pytest.approx(0.9)


def test_additive_data_fits_perfectly():
    f = F.free_wilson(C, Y)
    assert f["rss"] < 1e-20
    assert f["r_squared"] == pytest.approx(1.0)


def test_the_constraints_differ_in_coefficients():
    a = F.free_wilson(C, Y)["coefficients"]["intercept"]
    b = F.free_wilson(C, Y, "sum_zero")["coefficients"]["intercept"]
    assert abs(a - b) > 0.1


def test_but_agree_on_every_fitted_value():
    a = F.free_wilson(C, Y)["fitted"]
    b = F.free_wilson(C, Y, "sum_zero")["fitted"]
    assert all(x == pytest.approx(y, abs=1e-9) for x, y in zip(a, b))


def test_the_sum_zero_contributions_sum_to_zero():
    f = F.free_wilson(C, Y, "sum_zero")
    for p, grp in enumerate([P1, P2]):
        tot = sum(f["occurrences"]["P%d:%s" % (p + 1, g)]
                  * f["coefficients"]["P%d:%s" % (p + 1, g)]
                  for g in grp)
        assert tot == pytest.approx(0.0, abs=1e-9)


def test_prediction_adds_the_parts():
    f = F.free_wilson(C, Y)
    assert F.predict_activity(f, ("Br", "Et")) == pytest.approx(7.0)


def test_both_parameterisations_predict_alike():
    a = F.predict_activity(F.free_wilson(C, Y), ("Br", "Et"))
    b = F.predict_activity(F.free_wilson(C, Y, "sum_zero"),
                           ("Br", "Et"))
    assert a == pytest.approx(b, abs=1e-9)


def test_an_interaction_leaves_residuals():
    ny = list(Y)
    ny[C.index(("Br", "Et"))] += 2.0
    f = F.free_wilson(C, ny)
    assert f["rss"] > 0.5
    assert f["r_squared"] < 1.0


def test_occurrences_and_degrees_of_freedom_are_reported():
    f = F.free_wilson(C, Y)
    assert f["occurrences"]["P1:Cl"] == 3
    assert f["df_residual"] == len(C) - 5
    assert f["n_positions"] == 2


def test_the_design_matrix_names_its_columns():
    d = F.design_matrix(C)
    assert d["names"][0] == "intercept"
    assert len(d["names"]) == 5
    assert d["reference"] == ["H", "H"]
    assert all(len(r) == 5 for r in d["matrix"])


@pytest.mark.parametrize("call", [
    lambda: F.free_wilson([], []),
    lambda: F.free_wilson(C, Y[:3]),
    lambda: F.free_wilson([("H", "H"), ("H", "Me")], [1.0, 2.0]),
    lambda: F.free_wilson(C, Y, "ridge"),
    lambda: F.predict_activity(F.free_wilson(C, Y), ("Br",)),
    lambda: F.predict_activity(F.free_wilson(C, Y), ("I", "Me")),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
