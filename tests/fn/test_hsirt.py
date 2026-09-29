"""Tests for morie.fn.hsirt: first-order conditions of the item and scale likelihoods at the returned fit."""

import math

import pytest

from morie.fn.hsirt import heteroskedastic_irt

X = [-1.5 + 3 * i / 15 for i in range(16)]
V = [
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
]


def _Phi(z):
    # the model clips probabilities to [1e-9, 1 - 1e-9]
    return min(max(0.5 * math.erfc(-z / math.sqrt(2)), 1e-9), 1 - 1e-9)


def _phi(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def test_first_order_conditions():
    r = heteroskedastic_irt(V, X, max_iter=200)
    psi, a, b = r["psi"], r["alpha"], r["beta"]
    assert abs(sum(math.log(p) for p in psi)) < 1e-12
    for j in range(10):
        # probit score of item j in (alpha, beta) with z_i = (b x_i - a) / psi_i
        g = [0.0, 0.0]
        for i in range(16):
            z = (b[j] * X[i] - a[j]) / psi[i]
            p = _Phi(z)
            s = _phi(z) * (V[i][j] - p) / (p * (1 - p))
            g[0] += s * (-1 / psi[i])
            g[1] += s * X[i] / psi[i]
        g = [g[0] - a[j] / 25, g[1] - b[j] / 25]  # N(0, 5^2) item priors
        assert max(abs(v) for v in g) < 1e-6
    ll = sum(
        V[i][j] * math.log(_Phi((b[j] * X[i] - a[j]) / psi[i]))
        + (1 - V[i][j]) * math.log(1 - _Phi((b[j] * X[i] - a[j]) / psi[i]))
        for i in range(16)
        for j in range(10)
    )
    assert abs(r["loglik"] - ll) < 1e-10


def test_fixed_items_psi_maximises_each_voters_likelihood():
    a = [((j * 5) % 7 - 3) / 6 for j in range(10)]
    b = [(1 if j % 2 == 0 else -1) * (0.6 + ((j * 3) % 4) / 5) for j in range(10)]
    r = heteroskedastic_irt(V, X, item_params=(a, b), max_iter=1)
    gm = None
    raw = []
    for i in range(16):

        def nll(t, i=i):
            return t * t / 2 - sum(
                V[i][j] * math.log(_Phi((b[j] * X[i] - a[j]) / math.exp(t)))
                + (1 - V[i][j]) * math.log(1 - _Phi((b[j] * X[i] - a[j]) / math.exp(t)))
                for j in range(10)
            )

        grid = [-3 + 6 * k / 6000 for k in range(6001)]
        raw.append(math.exp(min(grid, key=nll)))
    gm = math.exp(sum(math.log(p) for p in raw) / 16)
    for got, want in zip(r["psi"], raw):
        assert abs(math.log(got) - math.log(want / gm)) < 2e-3


def test_validation():
    with pytest.raises(ValueError):
        heteroskedastic_irt([[2, 0, 1]] * 4, [0.0] * 4)
    with pytest.raises(ValueError):
        heteroskedastic_irt([[1, 0, 1]] * 4, [0.0] * 2)
