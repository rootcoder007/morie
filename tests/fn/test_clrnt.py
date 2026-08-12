"""Tests for clrnt (Wood, Houston & Hallifax 2017)."""

import math

from morie.fn.clrnt import (CONSTANTS, blood_from_plasma, clrnt,
                            fu_hepatocytes, fu_microsomes,
                            hepatic_clearance_prediction, observed_clint_u,
                            prediction_accuracy, scale_to_liver)


def _term(x):
    return 10.0 ** (0.072 * x * x + 0.067 * x - 1.126)


def test_binding_equations():
    for lp in (-1.0, 0.0, 2.0, 4.5):
        assert abs(fu_microsomes(lp, 1.0) -
                   1.0 / (1.0 + _term(lp))) < 1e-12
        assert abs(fu_hepatocytes(lp, 0.005) -
                   1.0 / (1.0 + 125.0 * 0.005 * _term(lp))) < 1e-12
    assert fu_microsomes(4.0) < fu_microsomes(2.0) < fu_microsomes(0.0)


def test_scaling_constants_are_the_papers():
    assert CONSTANTS["human"]["qh"] == 20.7
    assert CONSTANTS["rat"]["qh"] == 100.0
    assert CONSTANTS["human"]["liver_weight"] == 21.4
    assert CONSTANTS["rat"]["liver_weight"] == 40.0
    assert CONSTANTS["human"]["microsomes_pbsf"] == 40.0
    assert CONSTANTS["rat"]["microsomes_pbsf"] == 60.0
    assert CONSTANTS["human"]["hepatocytes_pbsf"] == 120e6
    assert abs(scale_to_liver(0.02, 0.5, "hepatocytes", "human") -
               0.02 * 120.0 * 21.4 / 0.5) < 1e-9
    assert abs(scale_to_liver(0.02, 0.5, "microsomes", "rat") -
               0.02 * 60.0 * 40.0 / 0.5) < 1e-9


def test_liver_models():
    assert abs(observed_clint_u(3.0, 0.1, "human") -
               3.0 / (0.1 * (1.0 - 3.0 / 20.7))) < 1e-9
    assert abs(observed_clint_u(3.0, 0.1, "human",
                                liver_model="parallel_tube") -
               (-20.7 * math.log(1.0 - 3.0 / 20.7) / 0.1)) < 1e-9
    lo = (observed_clint_u(0.5, 0.1) /
          observed_clint_u(0.5, 0.1, liver_model="parallel_tube"))
    hi = (observed_clint_u(18.0, 0.1) /
          observed_clint_u(18.0, 0.1, liver_model="parallel_tube"))
    assert abs(lo - 1.0) < 0.03 and hi > 2.0
    try:
        observed_clint_u(25.0, 0.1, "human")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_blood_from_plasma_defaults():
    _, _, rb = blood_from_plasma(10.0, 0.2, None, "acidic")
    assert rb == 0.55
    assert blood_from_plasma(10.0, 0.2, None, "basic")[2] == 1.0
    assert blood_from_plasma(10.0, 0.2, 0.8, "acidic")[2] == 0.8


def test_accuracy_statistics():
    pred = [10.0, 40.0, 5.0, 100.0]
    obs = [30.0, 120.0, 15.0, 300.0]
    acc = prediction_accuracy(pred, obs)
    assert abs(acc["afe"] - 1.0 / 3.0) < 1e-12
    assert abs(acc["fold_underprediction"] - 3.0) < 1e-12
    assert abs(acc["average_esf"] - 3.0) < 1e-12
    assert abs(acc["rmse"] - math.sqrt(sum((pred[i] - obs[i]) ** 2
                                           for i in range(4)) / 4.0)) < 1e-9
    assert acc["within_fold"] == 0.0
    perfect = prediction_accuracy([1.0, 2.0], [1.0, 2.0])
    assert perfect["afe"] == 1.0 and perfect["within_fold"] == 1.0


def test_end_to_end_recovers_a_known_shortfall():
    shortfall = 5.0
    clints = [0.002 * 1.7 ** k for k in range(8)]
    logps = [1.0 + 0.4 * k for k in range(8)]
    fus = [fu_hepatocytes(v) for v in logps]
    preds = [scale_to_liver(clints[i], fus[i]) for i in range(8)]
    fubs = [0.05 + 0.02 * (i % 3) for i in range(8)]
    clhs = []
    for i in range(8):
        target = shortfall * preds[i]
        clhs.append(target * fubs[i] * 20.7 / (20.7 + target * fubs[i]))
    res = clrnt(clint_in_vitro=clints, cl_h=clhs, fu_blood=fubs,
                log_pd=logps)
    assert abs(res["accuracy"]["fold_underprediction"] - shortfall) < 1e-6
    assert abs(res["accuracy"]["average_esf"] - shortfall) < 1e-6
    assert res["accuracy"]["within_fold"] == 0.0


def test_single_compound_and_measured_fu():
    single = clrnt(clint_in_vitro=0.02, log_pd=2.0)
    assert isinstance(single["predicted"], float)
    assert clrnt(clint_in_vitro=0.02,
                 fu_incubation=0.4)["fu_incubation"] == 0.4
    plasma = clrnt(clint_in_vitro=0.02, log_pd=2.0, cl_plasma=5.0,
                   fu_plasma=0.1, charge="acidic")
    assert abs(plasma["cl_h"] - 5.0 / 0.55) < 1e-12


def test_validation():
    for call in (lambda: scale_to_liver(1.0, 0.5, "hepatocytes", "mouse"),
                 lambda: scale_to_liver(1.0, 0.5, "slices", "human"),
                 lambda: scale_to_liver(1.0, 0.0),
                 lambda: observed_clint_u(3.0, 0.1, "human",
                                          liver_model="dispersion"),
                 lambda: observed_clint_u(-1.0, 0.1),
                 lambda: prediction_accuracy([1.0, 2.0], [1.0]),
                 lambda: clrnt(clint_in_vitro=0.02),
                 lambda: clrnt(clint_in_vitro=[0.02, 0.03], log_pd=[1.0])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert hepatic_clearance_prediction is clrnt
