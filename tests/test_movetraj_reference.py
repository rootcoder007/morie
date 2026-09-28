"""movetraj: trajectory measures by definition and random-walk models by their distributional properties."""

import math

import pytest

from morie.fn.movetraj import (
    first_passage_time,
    mean_squared_displacement,
    od_matrix,
    simulate_walk,
    trajectory_metrics,
)


def test_metrics_by_hand():
    m = trajectory_metrics([(0, 0), (3, 0), (3, 4), (0, 4)])
    assert m.steps == [3.0, 4.0, 3.0] and m.path_length == 10.0
    assert m.turning == pytest.approx([math.pi / 2, math.pi / 2])
    assert m.straightness == pytest.approx(0.4)
    p, c = 10 / 3, 0.0
    b = math.sqrt(((3 - p) ** 2 * 2 + (4 - p) ** 2) / 3) / p
    assert m.sinuosity == pytest.approx(2 / math.sqrt(p * ((1 + c) / (1 - c) + b * b)), abs=1e-15)
    # turning angles are wrapped: a left turn of 270 degrees is -90 degrees
    assert trajectory_metrics([(0, 0), (1, 0), (1, -1)]).turning == pytest.approx([-math.pi / 2])


def test_msd_and_fpt():
    assert mean_squared_displacement([(0, 0), (1, 1), (2, 2)], 1) == [2.0]
    f = first_passage_time([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], [0, 2, 4, 6, 8], 1.5)
    assert f[2] == pytest.approx(6.0) and all(v != v for i, v in enumerate(f) if i != 2)


def test_correlated_walk_sinuosity_and_turning_distribution():
    rho = 0.6
    w = simulate_walk(4000, kind="correlated", rho=rho, seed=5)
    m = trajectory_metrics(w.track)
    # wrapped Cauchy: E cos(turn) = rho; constant steps give S = 2 / sqrt((1 + rho)/(1 - rho))
    cbar = sum(math.cos(t) for t in m.turning) / len(m.turning)
    assert cbar == pytest.approx(rho, abs=0.03)
    assert m.sinuosity == pytest.approx(2 / math.sqrt((1 + cbar) / (1 - cbar)), abs=1e-12)


def test_levy_and_biased_walks():
    mu = 2.5
    lv = simulate_walk(20000, kind="levy", mu=mu, step=1.0, seed=6)
    # log step is exponential with mean 1 / (mu - 1)
    ml = sum(math.log(s) for s in lv.steps) / len(lv.steps)
    assert ml == pytest.approx(1 / (mu - 1), abs=0.02)
    assert min(lv.steps) >= 1.0
    bw = simulate_walk(400, kind="biased", target=(50.0, 0.0), bias=0.9, rho=0.3, seed=7)
    assert math.dist(bw.track[-1], (50.0, 0.0)) < 5.0


def test_od_matrix():
    r = od_matrix(["a", "a", "b", "c"], ["b", "b", "a", "a"], weights=[1, 2, 3, 4])
    assert r.zones == ["a", "b", "c"]
    assert r.matrix == [[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert r.production == [3.0, 3.0, 4.0] and r.attraction == [7.0, 3.0, 0.0]
    with pytest.raises(ValueError):
        simulate_walk(3, kind="brownian")
