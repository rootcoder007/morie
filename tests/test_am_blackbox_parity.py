"""Aldrich-McKelvey (closed form) and blackbox (EM low-rank) against the R arm.

The expected values are rmorie's morie_spatial_voting_aldrich_mckelvey()
and morie_spatial_voting_blackbox() (native path) on the same matrices.
"""

import math

import pytest

from morie._spatial_voting import aldrich_mckelvey, blackbox_scaling

NA = float("nan")
Z = [
    [1, 3, 4, 6],
    [2, 3, 5, 7],
    [1, 2, 2, 5],
    [3, 4, 6, 7],
    [1, NA, 4, 5],
    [2, 2, 3, 6],
    [4, 4, 4, 4],
    [1, 4, 5, 7],
]
X = [
    [1, 2, 3, 4],
    [2, 2, 4, 5],
    [5, 4, 2, 1],
    [4, 5, 1, 2],
    [3, NA, 3, 3],
    [1, 1, 5, 4],
    [2, 3, NA, 5],
    [5, 5, 1, 1],
]


def test_aldrich_mckelvey_matches_the_r_arm():
    r = aldrich_mckelvey(Z)
    expect = [-1.05900363487279, -0.412596097382783, 0.176571435268191, 1.29502829698738]
    assert [float(v) for v in r["zhat"]] == pytest.approx(expect, abs=1e-12)
    assert r["converged"] is True


def test_aldrich_mckelvey_minimises_its_loss():
    z = [float(v) for v in aldrich_mckelvey(Z)["zhat"]]

    def loss(v):
        tot = 0.0
        for row in Z:
            if any(math.isnan(t) for t in row) or len(set(row)) == 1:
                continue
            m = sum(row) / 4
            mv = sum(v) / 4
            sxx = sum((t - m) ** 2 for t in row)
            b = sum((row[j] - m) * (v[j] - mv) for j in range(4)) / sxx
            tot += sum((mv + b * (row[j] - m) - v[j]) ** 2 for j in range(4))
        return tot

    best = loss(z)
    for k in range(4):
        for eps in (-0.02, 0.02):
            w = list(z)
            w[k] += eps
            mw = sum(w) / 4
            sd = math.sqrt(sum((t - mw) ** 2 for t in w) / 3)
            w = [(t - mw) / sd for t in w]
            assert loss(w) >= best - 1e-12


def test_blackbox_matches_the_r_arm():
    r = blackbox_scaling(X, n_dims=1, minscale=3)
    fit_r = [
        -1.1275291738432, -0.984401234765863, 0.96062464488492, 1.08929093745305,
        -1.36662894098068, -1.19314980771862, 1.16433124000177, 1.32028204219072,
        1.64803299557076, 1.43883258492108, -1.40407995451917, -1.5921427563421,
        1.57757885954138, 1.37732185853758, -1.34405491838356, -1.52407795264278,
        0.0416121826101918, 0.0363299548189536, -0.0354524646192091, -0.040200976131219,
        -1.88072800396594, -1.64198941571713, 1.60232986679789, 1.81694630884927,
        -1.04455791316015, -0.911962300713212, 0.889935353930585, 1.00913350611757,
        2.15221999422763, 1.8790183606372, -1.83363376809323, -2.07923110949451,
    ]  # fmt: skip
    P = r["ideal_points"].tolist()
    W = r["stimuli_weights"].tolist()
    fit = [P[i][0] * W[j][0] for i in range(8) for j in range(4)]
    assert fit == pytest.approx(fit_r, abs=1e-10)
    assert float(r["singular_values"][0]) == pytest.approx(7.76123238157312, abs=1e-10)


def test_blackbox_leaves_thin_respondents_unscaled():
    r = blackbox_scaling(X, n_dims=1, minscale=4)
    P = r["ideal_points"].tolist()
    assert math.isnan(P[4][0]) and math.isnan(P[6][0])
    with pytest.raises(ValueError, match="two issues"):
        blackbox_scaling([[1.0], [2.0]], n_dims=1)
