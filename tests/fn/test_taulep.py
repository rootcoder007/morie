"""Tests for taulep (explicit tau-leaping).

Replaces the generated stub, which imported ``tau_leap_sim``.
"""

from morie.fn.taulep import taulep


def _decay(rate=0.1):
    # a single reaction X -> 0 with propensity rate * X
    nu = [[-1]]

    def propensity(x):
        return [rate * x[0]]

    return nu, propensity


def test_a_pure_decay_falls_toward_zero():
    nu, prop = _decay(0.2)
    res = taulep(nu, prop, [1000], tau=0.1, n_steps=200, seed=1)
    assert res["path"][-1][0] < res["path"][0][0]
    assert res["path"][-1][0] < 100


def test_the_population_never_goes_negative():
    nu, prop = _decay(0.5)
    res = taulep(nu, prop, [50], tau=0.5, n_steps=100, seed=2)
    assert all(state[0] >= 0 for state in res["path"])


def test_the_path_has_one_state_per_step_plus_the_start():
    nu, prop = _decay()
    res = taulep(nu, prop, [100], tau=0.1, n_steps=25, seed=1)
    assert len(res["path"]) == 26
    assert len(res["times"]) == 26
    assert abs(res["times"][-1] - 2.5) < 1e-9


def test_the_mean_follows_the_deterministic_solution():
    # E[X(t)] = X0 exp(-rate t); with many runs the average should track
    rate, x0, t = 0.1, 5000, 5.0
    nu, prop = _decay(rate)
    finals = []
    for seed in range(8):
        res = taulep(nu, prop, [x0], tau=0.05, n_steps=100, seed=seed)
        finals.append(res["path"][-1][0])
    mean = sum(finals) / len(finals)
    import math
    want = x0 * math.exp(-rate * t)
    assert abs(mean - want) / want < 0.1


def test_total_firings_are_recorded_per_reaction():
    # one entry per reaction channel, summed over the run
    nu, prop = _decay(0.3)
    res = taulep(nu, prop, [200], tau=0.2, n_steps=10, seed=3)
    assert len(res["firings"]) == 1
    assert res["firings"][0] >= 0
    # every firing removes one molecule, so they account for the drop
    assert res["firings"][0] == 200 - res["path"][-1][0]


def test_seed_reproducibility():
    nu, prop = _decay()
    a = taulep(nu, prop, [100], tau=0.1, n_steps=20, seed=9)["path"]
    b = taulep(nu, prop, [100], tau=0.1, n_steps=20, seed=9)["path"]
    assert a == b


def test_validation():
    nu, prop = _decay()
    for call in (lambda: taulep([[-1, 1]], prop, [10], 0.1, 5),
                 lambda: taulep(nu, prop, [10], 0.0, 5),
                 lambda: taulep(nu, lambda x: [-1.0], [10], 0.1, 5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
