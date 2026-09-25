"""Tests for bnshrt: bounds in a short dynamic binary-choice panel.

Honore and Tamer (2006), eq. (2): y_it = 1{x_it'b + y_i,t-1 g + a_i + e_it >= 0}
with the heterogeneity a_i and the initial state y_i0 left unrestricted.
A parameter value is in the identified set when the observed sequence
frequencies lie in the convex hull of the model's sequence probabilities.
"""

import itertools
import math

import pytest

from morie.fn.bnshrt import (bound_short_panel, identified_set,
                             in_identified_set, sequence_frequencies,
                             sequence_probabilities)

X = [[0.5], [-0.2], [0.3]]           # one covariate over T = 3 periods
BETA, GAMMA = 0.8, 0.6
SUPPORT = [(-1.0, 0, 0.4), (1.2, 1, 0.6)]   # (alpha, y0, mixing weight)


def _logistic(z):
    return 1.0 / (1.0 + math.exp(-z))


def _population_freq():
    freq = {}
    for a, y0, w in SUPPORT:
        for seq, p in sequence_probabilities([BETA], GAMMA, X, a, y0).items():
            freq[seq] = freq.get(seq, 0.0) + w * p
    return freq


def test_bnshrt_basic():
    """Sequence probabilities are a distribution and match the model."""
    for a, y0, _ in SUPPORT:
        sp = sequence_probabilities([BETA], GAMMA, X, a, y0)
        assert len(sp) == 2 ** len(X)
        assert sum(sp.values()) == pytest.approx(1.0, abs=1e-12)
        # every sequence probability is the product of the per-period
        # logistic choice probabilities, recomputed here by hand
        for seq, p in sp.items():
            prev, want = y0, 1.0
            for t, y in enumerate(seq):
                q = _logistic(X[t][0] * BETA + GAMMA * prev + a)
                want *= q if y == 1 else 1.0 - q
                prev = y
            assert p == pytest.approx(want, rel=1e-12)

    # one period: P(y = 1) is the logistic index itself
    one = sequence_probabilities([BETA], GAMMA, [[0.5]], 0.3, 1)
    assert one[(1,)] == pytest.approx(_logistic(0.5 * BETA + GAMMA + 0.3),
                                      rel=1e-12)

    Y = [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]]
    f = sequence_frequencies(Y)
    assert f[(0, 1, 1)] == 0.5 and f[(1, 0, 0)] == 0.25
    assert f[(1, 1, 1)] == 0.25 and f[(0, 0, 0)] == 0.0
    assert sum(f.values()) == pytest.approx(1.0, abs=1e-15)


def test_true_parameter_is_feasible_and_a_distant_one_is_not():
    freq = _population_freq()
    alpha_grid = [-1.0, 1.2]            # contains the true support
    ok = in_identified_set(freq, [BETA], GAMMA, X, alpha_grid)
    assert ok["feasible"]
    assert ok["discrepancy"] < 1e-6
    # the recovered weights put the true mass on the true support points
    w = ok["weights"]                   # order: (a, y0) over grid x {0, 1}
    assert w[0] == pytest.approx(0.4, abs=1e-4)   # alpha=-1, y0=0
    assert w[3] == pytest.approx(0.6, abs=1e-4)   # alpha=1.2, y0=1
    bad = in_identified_set(freq, [-3.0], -2.5, X, alpha_grid)
    assert not bad["feasible"]


def test_identified_set_contains_the_truth():
    freq = _population_freq()
    N = 200000
    Y = []
    for seq, p in freq.items():
        Y += [list(seq)] * int(round(p * N))
    res = bound_short_panel(Y, X, [0.0, BETA, 2.5], [-1.5, GAMMA, 2.0],
                            [-1.0, 1.2])
    assert (BETA, GAMMA) in res["set"]
    assert identified_set is bound_short_panel


def test_bnshrt_edge():
    """Each input check fires."""
    with pytest.raises(ValueError, match="0/1"):
        sequence_frequencies([[0, 2, 1]])
    with pytest.raises(ValueError, match="same length"):
        sequence_frequencies([[0, 1, 1], [1, 0]])
    with pytest.raises(ValueError, match="no observations"):
        sequence_frequencies([])
    with pytest.raises(ValueError, match="link"):
        sequence_probabilities([BETA], GAMMA, X, 0.0, 0, link="cauchit")
    with pytest.raises(ValueError, match="covariates"):
        sequence_probabilities([BETA, 1.0], GAMMA, X, 0.0, 0)
    with pytest.raises(ValueError, match="alpha grid"):
        in_identified_set(_population_freq(), [BETA], GAMMA, X, [])
