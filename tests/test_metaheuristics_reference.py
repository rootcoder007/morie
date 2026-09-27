"""Population metaheuristics on Philox: optimum found, monotone best-so-far history, bounds, and cross-arm pins."""

import pytest

from morie.fn.abcopt import artificial_bee_colony
from morie.fn.acorop import ant_colony_continuous
from morie.fn.batalg import bat_algorithm
from morie.fn.cuckoo import cuckoo_search
from morie.fn.fflyop import firefly_algorithm
from morie.fn.gwoopt import grey_wolf_optimizer
from morie.fn.hhoopt import harris_hawks_optimizer
from morie.fn.jayaop import jaya_algorithm
from morie.fn.mfoopt import moth_flame_optimizer
from morie.fn.scaopt import sine_cosine_algorithm
from morie.fn.tlbopt import teaching_learning_optimizer
from morie.fn.woaopt import whale_optimization

ALL = [
    artificial_bee_colony,
    ant_colony_continuous,
    bat_algorithm,
    cuckoo_search,
    firefly_algorithm,
    grey_wolf_optimizer,
    harris_hawks_optimizer,
    jaya_algorithm,
    moth_flame_optimizer,
    sine_cosine_algorithm,
    teaching_learning_optimizer,
    whale_optimization,
]
PINS = {
    "artificial_bee_colony": [0.05559464590320143, 972, [0.774017089863577, 0.6058302743257891, 0.5001244220202218]],
    "ant_colony_continuous": [
        0.30495372509955965,
        492,
        [0.45320906041240255, 0.19768811021688312, 0.49466560548144406],
    ],
    "bat_algorithm": [0.08739844589898266, 492, [0.9578377366073769, 0.8886297532490737, 0.5504191096638418]],
    "cuckoo_search": [0.29117355519532445, 972, [1.0264899327737098, 1.0986571479630434, 0.20302914947941417]],
    "firefly_algorithm": [0.01764155556107064, 2519, [1.0624011758958554, 1.139558902914149, 0.5441355246779651]],
    "grey_wolf_optimizer": [0.06904140666355973, 492, [0.739858596570978, 0.5457486306128039, 0.4668608448719412]],
    "harris_hawks_optimizer": [0.07814798490439336, 712, [1.0215285230415343, 1.043188092277144, 0.22130041880954388]],
    "jaya_algorithm": [0.1499350784261021, 492, [0.6132871903110704, 0.3749232011925473, 0.515645013693825]],
    "moth_flame_optimizer": [0.21288260214321233, 492, [0.5387219297940847, 0.2892308096719295, 0.5026520721384228]],
    "sine_cosine_algorithm": [0.16712401479452704, 492, [0.5951045846525763, 0.35069165833217014, 0.544587818649549]],
    "teaching_learning_optimizer": [
        0.006392011790791555,
        972,
        [0.9654104755750363, 0.9305856064318236, 0.4293559824370671],
    ],
    "whale_optimization": [2.186490625843569, 492, [-0.4118431411891823, 0.17301631505753426, 0.9382151545143167]],
}


def rb3(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + (x[2] - 0.5) ** 2


def sphere(x):
    return sum(v * v for v in x)


@pytest.mark.parametrize("alg", ALL, ids=lambda a: a.__name__)
def test_best_so_far_is_consistent_and_in_bounds(alg):
    bounds = [(-5, 5), (-4, 6), (-3, 3)]
    r = alg(sphere, bounds, n_pop=10, max_iter=30, seed=3)
    assert all(a >= b for a, b in zip(r["history"], r["history"][1:]))
    assert r["fun"] == r["estimate"] == sphere(r["x"]) == r["history"][-1]
    assert all(lo <= v <= hi for v, (lo, hi) in zip(r["x"], bounds))
    assert len(r["history"]) == 30
    assert r["fun"] < sphere([0.5 * (lo + hi) + 1.0 for lo, hi in bounds])
    again = alg(sphere, bounds, n_pop=10, max_iter=30, seed=3)
    assert again["x"] == r["x"] and again["history"] == r["history"]
    assert alg(sphere, bounds, n_pop=10, max_iter=30, seed=4)["history"] != r["history"]


@pytest.mark.parametrize("alg", ALL, ids=lambda a: a.__name__)
def test_runs_match_the_r_arm(alg):
    r = alg(rb3, [(-2, 2), (-2, 2), (-1, 3)], n_pop=12, max_iter=40, seed=7)
    fun, nfev, x = PINS[alg.__name__]
    assert r["n_fev"] == nfev
    assert abs(r["fun"] - fun) <= 1e-12 * max(1.0, abs(fun))
    assert all(abs(a - b) <= 1e-12 * max(1.0, abs(b)) for a, b in zip(r["x"], x))


def test_sphere_is_solved_by_the_strong_methods():
    for alg in (
        artificial_bee_colony,
        ant_colony_continuous,
        grey_wolf_optimizer,
        harris_hawks_optimizer,
        teaching_learning_optimizer,
        whale_optimization,
    ):
        assert alg(sphere, [(-5, 5)] * 3, max_iter=300)["fun"] < 1e-20


def test_evaluation_counts_follow_the_algorithms():
    b = [(-5, 5)] * 2
    for alg in (
        ant_colony_continuous,
        bat_algorithm,
        grey_wolf_optimizer,
        jaya_algorithm,
        moth_flame_optimizer,
        sine_cosine_algorithm,
        whale_optimization,
    ):
        assert alg(sphere, b, n_pop=8, max_iter=5)["n_fev"] == 8 + 8 * 5
    for alg in (cuckoo_search, teaching_learning_optimizer):
        assert alg(sphere, b, n_pop=8, max_iter=5)["n_fev"] == 8 + 2 * 8 * 5


def test_bad_bounds_are_rejected():
    for alg in ALL:
        with pytest.raises(ValueError):
            alg(sphere, [(1, 1)])
