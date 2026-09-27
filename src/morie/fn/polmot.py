# morie.fn -- function file (rootcoder007/morie)
"""Equilibrium of two policy-motivated candidates under electoral uncertainty."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import solve, ssum
from .bfgsmin import bfgs_minimize


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _win_index(x1, x2, mu, sd):
    """z with P(candidate 2 wins) = Phi(z); gradients of z in x1 and x2."""
    d = [b - a for a, b in zip(x1, x2)]
    m = [(a + b) / 2.0 for a, b in zip(x1, x2)]
    r = math.sqrt(ssum(t * t for t in d))
    num = ssum((u - c) * t for u, c, t in zip(mu, m, d))
    z = num / (sd * r)
    g2 = [(u - b) / (sd * r) - num * t / (sd * r**3) for u, b, t in zip(mu, x2, d)]
    g1 = [(a - u) / (sd * r) + num * t / (sd * r**3) for u, a, t in zip(mu, x1, d)]
    return z, g1, g2


def _loss(y, a):
    return -ssum((p - q) ** 2 for p, q in zip(y, a))


def expected_utilities(x1, x2, a1, a2, mu, sd):
    z, _, _ = _win_index(x1, x2, mu, sd)
    p2 = _cdf(z)
    return (1 - p2) * _loss(x1, a1) + p2 * _loss(x2, a1), p2 * _loss(x2, a2) + (1 - p2) * _loss(x1, a2), p2


def policy_motivated_equilibrium(
    ideal_1, ideal_2, center, sd, *, start=None, tol: float = 1e-10, max_iter: int = 500
) -> DescriptiveResult:
    """Nash equilibrium of Wittman's policy-motivated two-candidate game.

    Candidates k = 1, 2 have quadratic losses ``u_k(y) = -||y - a_k||^2``
    and care only about the policy that wins. The decisive voter sits at
    ``m ~ N(center, sd^2 I)`` and elects the nearer platform, so candidate 2
    wins with ``P = Phi(((center - (x1 + x2)/2)' (x2 - x1)) / (sd ||x2 - x1||))``
    (Calvert 1985). Candidate k maximises ``P_k u_k(x_k) + (1 - P_k) u_k(x_other)``.
    The equilibrium is found by alternating best responses (BFGS on the
    analytic gradients) from ``start`` (default: each ideal point pulled towards the centre,
    ``c + t (a_k - c)`` with ``t = sd / (sd + ||a_k - c||)``), then
    polished by Newton steps on the joint first-order conditions. In one
    dimension with ``a_1 = -a``, ``a_2 = a`` and ``center = 0`` the platforms
    are ``+-a / (1 + 2 a phi(0) / sd)``: divergent, and converging to the
    centre as ``sd`` shrinks, Calvert's certainty result.

    :param ideal_1: Ideal point of candidate 1 (sequence of length d).
    :param ideal_2: Ideal point of candidate 2.
    :param center: Mean of the decisive voter's position.
    :param sd: Standard deviation of each coordinate of that position (> 0).
    :param start: Optional starting platforms ``(x1, x2)``.
    :param tol: Stop when no platform moves more than ``tol``.
    :param max_iter: Maximum best-response rounds.
    :return: DescriptiveResult; ``value`` is ``[x1, x2]``; ``extra`` has
        ``win_prob_2``, ``utilities``, ``gradients`` (of each candidate's
        expected utility in its own platform), ``rounds`` and ``converged``.

    References
    ----------
    Wittman, D. (1983). Candidate motivation: a synthesis of alternative
    theories. American Political Science Review 77, 142-157.

    Calvert, R. L. (1985). Robustness of the multidimensional voting model:
    candidate motivations, uncertainty, and convergence. American Journal of
    Political Science 29, 69-95.

    Examples
    --------
    >>> r = policy_motivated_equilibrium([-1.0], [1.0], [0.0], 0.5)
    >>> [round(p[0], 9) for p in r.value]
    [-0.385242274, 0.385242274]
    >>> round(1 / (1 + 2 * 0.3989422804014327 / 0.5), 9)
    0.385242274
    """
    a1 = [float(t) for t in ideal_1]
    a2 = [float(t) for t in ideal_2]
    mu = [float(t) for t in center]
    sd = float(sd)
    if sd <= 0:
        raise ValueError("sd must be positive")
    if start is not None:
        x1, x2 = [list(map(float, p)) for p in start]
    else:

        def shrink(a):
            dist = math.sqrt(ssum((p - q) ** 2 for p, q in zip(a, mu)))
            t = sd / (sd + dist)
            return [q + t * (p - q) for p, q in zip(a, mu)]

        x1, x2 = shrink(a1), shrink(a2)

    def grads(p1, p2):
        z, g1, g2 = _win_index(p1, p2, mu, sd)
        p = _cdf(z)
        f = _phi(z)
        d1 = f * (_loss(p1, a1) - _loss(p2, a1))
        d2 = f * (_loss(p2, a2) - _loss(p1, a2))
        G1 = [-d1 * g - 2 * (1 - p) * (u - a) for g, u, a in zip(g1, p1, a1)]
        G2 = [d2 * g - 2 * p * (u - a) for g, u, a in zip(g2, p2, a2)]
        return G1, G2

    rounds, converged = 0, False
    while rounds < max_iter:
        rounds += 1
        b2 = bfgs_minimize(
            lambda y, x1=x1: -expected_utilities(x1, y, a1, a2, mu, sd)[1],
            x2,
            grad=lambda y, x1=x1: [-g for g in grads(x1, y)[1]],
            gtol=1e-12,
        )["x"]
        b1 = bfgs_minimize(
            lambda y, b2=b2: -expected_utilities(y, b2, a1, a2, mu, sd)[0],
            x1,
            grad=lambda y, b2=b2: [-g for g in grads(y, b2)[0]],
            gtol=1e-12,
        )["x"]
        move = max(abs(p - q) for p, q in zip(b1 + b2, x1 + x2))
        x1, x2 = list(b1), list(b2)
        if move <= tol:
            converged = True
            break
    d = len(a1)

    def foc(w):
        g1, g2 = grads(w[:d], w[d:])
        return g1 + g2

    w = x1 + x2
    for _ in range(20):
        F = foc(w)
        if max(abs(t) for t in F) <= 1e-13:
            break
        h = 1e-6
        J = [[0.0] * (2 * d) for _ in range(2 * d)]
        for c in range(2 * d):
            wp, wm = list(w), list(w)
            wp[c] += h
            wm[c] -= h
            Fp, Fm = foc(wp), foc(wm)
            for r in range(2 * d):
                J[r][c] = (Fp[r] - Fm[r]) / (2 * h)
        step = solve(J, [-t for t in F])
        w = [a + b for a, b in zip(w, step)]
    x1, x2 = w[:d], w[d:]
    u1, u2, p2 = expected_utilities(x1, x2, a1, a2, mu, sd)
    G1, G2 = grads(x1, x2)
    return DescriptiveResult(
        name="policy_motivated_equilibrium",
        value=[x1, x2],
        extra={
            "win_prob_2": p2,
            "utilities": [u1, u2],
            "gradients": [G1, G2],
            "rounds": rounds,
            "converged": converged,
        },
    )


polmot = policy_motivated_equilibrium


def cheatsheet() -> str:
    return "policy_motivated_equilibrium(a1, a2, center, sd) -> Wittman-Calvert policy-motivated equilibrium"
