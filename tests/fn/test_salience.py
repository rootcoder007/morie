"""salience: L-moments, Schofield's coefficient and the issue-by-issue median from their definitions."""

import math
from itertools import combinations

import pytest

from morie.fn._rng import random_normal, random_uniform
from morie.fn.salience import attention_punctuation, structure_induced_equilibrium, valence_convergence


def test_l_moments_from_pairs_and_quads():
    u = [float(v) for v in random_uniform(40, seed=3)]
    ser = [[1 + 5 * u[8 * k + t] for t in range(8)] for k in range(3)]
    r = attention_punctuation(ser)
    x = sorted(r.changes)
    n = len(x)
    # l2 and l4 as averages over all subsamples of order statistics (Hosking 1990, eq 2.2)
    l2 = sum(b - a for a, b in combinations(x, 2)) / 2 / math.comb(n, 2)
    l4 = sum(d - 3 * c + 3 * b - a for a, b, c, d in combinations(x, 4)) / 4 / math.comb(n, 4)
    assert r.l_moments[1] == pytest.approx(l2, rel=1e-12)
    assert r.l_kurtosis == pytest.approx(l4 / l2, rel=1e-10)
    m = sum(x) / n
    assert r.kurtosis == pytest.approx(
        sum((v - m) ** 4 for v in x) / n / (sum((v - m) ** 2 for v in x) / n) ** 2, rel=1e-12
    )


def test_valence_convergence():
    z = [float(v) for v in random_normal(40, seed=6)]
    X = [[z[i], 0.5 * z[20 + i]] for i in range(20)]
    lam = [0.0, 0.8, 1.6]
    r = valence_convergence(X, lam, 0.7)
    rho1 = 1 / (1 + math.exp(0.8) + math.exp(1.6))
    mx = sum(p[0] for p in X) / 20
    my = sum(p[1] for p in X) / 20
    nu2 = sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in X) / 20
    assert r.c == pytest.approx(2 * 0.7 * (1 - 2 * rho1) * nu2, rel=1e-12)
    assert sum(r.eigenvalues) == pytest.approx(r.c - 2, rel=1e-12)
    assert r.local_equilibrium == (max(r.eigenvalues) < 0)
    equal = valence_convergence(X, [1.0, 1.0], 0.7)
    assert pytest.approx(0.0, abs=1e-15) == equal.A and equal.local_equilibrium


def test_structure_induced_equilibrium():
    X = [[3.0, -1.0], [1.0, 4.0], [2.0, 0.0], [5.0, 2.0], [0.0, 1.0]]
    assert structure_induced_equilibrium(X) == [2.0, 1.0]
    assert structure_induced_equilibrium(X, [5.0, 1.0, 1.0, 1.0, 1.0]) == [3.0, -1.0]
