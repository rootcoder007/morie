"""Morin (2016) eqs 3.94, 4.77-4.80, 6.16-6.18, 6.28-6.34: exhaustive enumeration, series, round trips, normalisation."""

import itertools
import math

from morie.fn.bvnmodel import bvnmodel
from morie.fn.geomexp import geomexp
from morie.fn.linmodel import linmodel
from morie.fn.linmodelinv import linmodelinv
from morie.fn.sampvarvar import sampvarvar


def test_var_of_sample_variance_by_enumeration():
    die = [1, 2, 3, 4, 5, 6]
    for n in (2, 3, 4):
        s2 = []
        for draw in itertools.product(die, repeat=n):  # all 6^n equally likely samples
            m = sum(draw) / n
            s2.append(sum((v - m) ** 2 for v in draw) / (n - 1))
        mean = sum(s2) / len(s2)
        var = sum((v - mean) ** 2 for v in s2) / len(s2)
        r = sampvarvar(die, [1 / 6] * 6, n)
        assert abs(mean - 35 / 12) < 1e-12  # E[s^2] = sigma^2 = 2.92 (3.74)
        assert abs(r["var_s2"] - var) < 1e-12
    assert abs(sampvarvar(die, [1 / 6] * 6, 2)["var_s2"] - 1673 / 144) < 1e-12  # (mu4 + sigma^4)/2, exact
    # Bernoulli(1/2): mu4 = sigma^4 = 1/16 -> Var(s^2) = (1/16)(1 - (n-3)/(n-1))/n
    assert abs(sampvarvar([0, 1], [0.5, 0.5], 5)["var_s2"] - (1 / 16) * (1 - 2 / 4) / 5) < 1e-15


def test_geometric_expectation():
    for p in (0.25, 1 / 6, 0.9, 1.0):
        r = geomexp(p)
        assert abs(r["series"] - 1 / p) < 1e-12 and abs(r["row_sums"] - 1 / p) < 1e-12 and r["mean"] == 1 / p
    assert abs(geomexp(0.5, terms=3)["series"] - (0.5 + 2 * 0.25 + 3 * 0.125)) < 1e-15


def test_linear_model_round_trip():
    f = linmodel(m=1.0, sigma_x=7.5, sigma_z=10.6)
    b = linmodelinv(7.5, f["sigma_y"], f["r"])
    assert abs(b["m"] - 1.0) < 1e-12 and abs(b["sigma_z"] - 10.6) < 1e-12
    assert abs(f["r"] - 7.5 / math.sqrt(7.5**2 + 10.6**2)) < 1e-15  # the book's 0.58 (6.77)
    b = linmodelinv(2.0, 3.0, -0.4)
    assert abs(b["m"] + 0.6) < 1e-15 and abs(b["sigma_z"] - 3 * math.sqrt(0.84)) < 1e-15


def test_joint_density_forms_and_normalisation():
    for x, y in ((0.3, -1.2), (1.5, 2.0), (-2.0, 0.1)):
        r = bvnmodel(x, y, 0.8, 1.3, 0.6)
        phi = lambda t, s: math.exp(-t * t / (2 * s * s)) / (s * math.sqrt(2 * math.pi))  # noqa: E731
        assert abs(r["density"] - phi(x, 1.3) * phi(y - 0.8 * x, 0.6)) < 1e-15  # (6.28)
        assert abs(r["density"] - r["density_r"]) < 1e-14  # (6.30) = (6.31)
    h = 0.03  # midpoint rule over +-9 (about 7 sd in x and 7.6 in y)
    total = sum(
        bvnmodel(-9 + (i + 0.5) * h, -9 + (j + 0.5) * h, 0.8, 1.3, 0.6)["density"] * h * h
        for i in range(600)
        for j in range(600)
    )
    assert abs(total - 1) < 1e-6  # (6.34): integrates to one
