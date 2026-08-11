"""Tests for thomp.thompson_sampling.

Anchors: the conjugate Beta-Bernoulli posterior recomputed by
independent counting (Russo et al 2018, Algorithm 3.2 update line),
plus dominance and degenerate-prior behaviour.
"""

from morie.fn.thomp import thomp


def test_thomp_posterior_counts_are_exact_conjugate_updates():
    r = thomp([0.7, 0.3], 60, seed=17)
    # recount successes/failures per arm from the returned history
    succ = [0.0, 0.0]
    fail = [0.0, 0.0]
    for t in range(60):
        k = int(r["actions"][t])
        if r["rewards"][t] == 1.0:
            succ[k] += 1.0
        else:
            fail[k] += 1.0
    for k in range(2):
        assert r["alpha"][k] == 1.0 + succ[k]
        assert r["beta"][k] == 1.0 + fail[k]
        n = r["alpha"][k] + r["beta"][k]
        assert abs(r["post_mean"][k] - r["alpha"][k] / n) < 1e-16


def test_thomp_deterministic_rewards_with_unit_probs():
    # p = (1, 0): every play of arm 0 pays 1, of arm 1 pays 0, exactly.
    r = thomp([1.0, 0.0], 40, seed=3)
    assert r["total_reward"] == r["counts"][0]
    assert r["alpha"][0] == 1.0 + r["counts"][0]
    assert r["beta"][1] == 1.0 + r["counts"][1]


def test_thomp_overwhelming_prior_pins_first_arm():
    # Beta(1000, 1) on arm 0 vs Beta(1, 1000) on arm 1: theta_0 is
    # essentially 1 and theta_1 essentially 0 every period.
    r = thomp([0.5, 0.5], 25, alpha0=[1000.0, 1.0],
              beta0=[1.0, 1000.0], seed=9)
    assert r["counts"][0] == 25.0


def test_thomp_concentrates_on_better_arm():
    r = thomp([0.9, 0.1], 300, seed=42)
    assert r["estimate"] == 0.0
    assert r["counts"][0] > 200.0


def test_thomp_rejects_bad_probs():
    try:
        thomp([0.5, 1.2], 10)
    except ValueError:
        return
    raise AssertionError("p > 1 accepted")
