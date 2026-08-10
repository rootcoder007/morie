"""Tests for mdpval.mdp_value_iteration.

Anchor: exact rational-arithmetic solve of the Bellman equations by
deterministic-policy enumeration (fractions.Fraction Gaussian
elimination), a fully independent route from the iterative sweeps.
"""

from fractions import Fraction

from morie.fn.mdpval import mdpval

# 3-state, 2-action MDP; state 2 absorbing with zero reward.
P = [
    [[0.7, 0.3, 0.0], [0.1, 0.6, 0.3], [0.0, 0.0, 1.0]],
    [[0.2, 0.5, 0.3], [0.0, 0.4, 0.6], [0.0, 0.0, 1.0]],
]
R = [[0.5, 1.0], [-0.2, 0.8], [0.0, 0.0]]
GAMMA = 0.9


def _solve_policy(pol):
    """Exact V^pi from (I - gamma P_pi) V = R_pi with Fractions."""
    g = Fraction(9, 10)
    Pf = [[Fraction(x).limit_denominator(10**6) for x in P[pol[s]][s]]
          for s in range(3)]
    Rf = [Fraction(R[s][pol[s]]).limit_denominator(10**6) for s in range(3)]
    A = [[(Fraction(1) if i == j else Fraction(0)) - g * Pf[i][j]
          for j in range(3)] + [Rf[i]] for i in range(3)]
    for col in range(3):
        piv = next(r for r in range(col, 3) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        A[col] = [x / A[col][col] for x in A[col]]
        for r in range(3):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    return [A[i][3] for i in range(3)]


def _brute_optimal():
    best = None
    for a0 in range(2):
        for a1 in range(2):
            for a2 in range(2):
                V = _solve_policy([a0, a1, a2])
                if best is None or all(v >= b for v, b in zip(V, best)):
                    best = V
    return best


def test_mdpval_closed_form_two_state():
    # stay in 0 with reward 1 forever: V*(0) = 1 / (1 - gamma) exactly.
    P2 = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    R2 = [[1.0, 0.0], [0.0, 0.0]]
    r = mdpval(P2, R2, 0.5, tol=1e-14)
    assert abs(r["estimate"][0] - 2.0) < 1e-12
    assert abs(r["estimate"][1] - 0.0) < 1e-15
    assert r["policy"][0] == 0.0
    assert r["converged"]


def test_mdpval_matches_exact_policy_enumeration():
    exact = _brute_optimal()
    r = mdpval(P, R, GAMMA, tol=1e-13)
    for s in range(3):
        assert abs(r["estimate"][s] - float(exact[s])) < 1e-9
    # greedy policy must be optimal for the exact V as well
    q = r["q"]
    for s in range(3):
        b = int(r["policy"][s])
        for a in range(2):
            assert q[s, b] >= q[s, a] - 1e-12


def test_mdpval_gamma_zero_is_myopic():
    r = mdpval(P, R, 0.0, tol=1e-14)
    for s in range(3):
        assert abs(r["estimate"][s] - max(R[s])) < 1e-15


def test_mdpval_rejects_bad_rows():
    try:
        mdpval([[[0.5, 0.4], [0.0, 1.0]]], [[0.0], [0.0]], 0.9)
    except ValueError:
        return
    raise AssertionError("non-stochastic row accepted")
