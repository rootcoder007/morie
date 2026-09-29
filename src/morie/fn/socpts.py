# morie.fn -- function file (rootcoder007/morie)
"""Second-order cone programming."""

from __future__ import annotations

import math

from ._qpcore import solve
from ._richresult import RichResult

__all__ = ["second_order_cone"]


def _vec(v):
    v = v.tolist() if hasattr(v, "tolist") else v
    return [float(t) for t in v]


def _mat(A):
    A = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in A]


def _barrier(x, cones):
    """Barrier value, gradient and Hessian of ``-sum log(u_i^2 - |w_i|^2)``; None outside the interior."""
    n = len(x)
    val = 0.0
    g = [0.0] * n
    H = [[0.0] * n for _ in range(n)]
    for A, b, c, d in cones:
        u = sum(a * t for a, t in zip(c, x)) + d
        w = [sum(a * t for a, t in zip(r, x)) + bb for r, bb in zip(A, b)]
        q = u * u - sum(v * v for v in w)
        if u <= 0.0 or q <= 0.0:
            return None
        val -= math.log(q)
        dq = [2.0 * u * c[k] - 2.0 * sum(A[i][k] * w[i] for i in range(len(w))) for k in range(n)]
        for k in range(n):
            g[k] -= dq[k] / q
            for q2 in range(n):
                d2 = 2.0 * c[k] * c[q2] - 2.0 * sum(A[i][k] * A[i][q2] for i in range(len(w)))
                H[k][q2] += dq[k] * dq[q2] / (q * q) - d2 / q
    return val, g, H


def _centering(f, x, cones, t, max_newton=100):
    """Minimise ``t f'x + barrier`` by damped Newton with backtracking (Boyd and Vandenberghe alg. 9.5)."""
    n = len(x)
    for _ in range(max_newton):
        bv = _barrier(x, cones)
        val = t * sum(a * b for a, b in zip(f, x)) + bv[0]
        g = [t * f[k] + bv[1][k] for k in range(n)]
        H = [row[:] for row in bv[2]]
        for k in range(n):
            H[k][k] += 1e-14 * max(1.0, abs(H[k][k]))
        dx = [-v for v in solve(H, g)]
        lam2 = -sum(a * b for a, b in zip(g, dx))
        if lam2 / 2.0 <= 1e-12:
            break
        s = 1.0
        while True:
            xn = [a + s * b for a, b in zip(x, dx)]
            bn = _barrier(xn, cones)
            if bn is not None and t * sum(a * b for a, b in zip(f, xn)) + bn[0] <= val - 0.25 * s * lam2:
                break
            s *= 0.5
            if s < 1e-20:
                return x
        x = xn
    return x


def second_order_cone(c, A, b, domains, x0=None, eps=1e-10):
    r"""Second-order cone programming.

    Solves ``min c'x`` subject to ``||A_i x + b_i||_2 <= e_i'x + d_i`` for
    each cone ``i``, the cone's affine bound ``(e_i, d_i)`` given in
    ``domains`` (Lobo et al. 1998). Barrier method (Boyd and Vandenberghe
    2004, sec. 11.3): with the generalised logarithm ``-log((e_i'x +
    d_i)^2 - ||A_i x + b_i||^2)`` of each cone, minimise ``t c'x +
    barrier`` by damped Newton steps with backtracking, multiplying ``t``
    by 10 until the duality-gap bound ``m / t`` falls below ``eps``. A
    strictly feasible start is found by the phase I problem ``min s``
    subject to ``||A_i x + b_i|| <= e_i'x + d_i + s`` from ``x0`` (zeros).
    A purely linear constraint is the case ``A_i = 0``.

    Parameters
    ----------
    c : array-like, shape (n,)
        Objective.
    A : sequence of (k_i, n) matrices
        Cone matrices.
    b : sequence of length-k_i vectors
        Cone offsets.
    domains : sequence of (e_i, d_i)
        Affine right-hand sides.
    x0 : array-like, optional
        Phase I start.
    eps : float
        Duality-gap tolerance.

    Returns
    -------
    RichResult
        ``x``, ``objective``, ``lhs`` (norms), ``rhs``, ``slack``,
        ``feasible``, ``estimate`` (= objective).

    References
    ----------
    Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press, ch. 11.

    Lobo, M. S., Vandenberghe, L., Boyd, S. and Lebret, H. (1998). Applications of second-order cone
    programming. *Linear Algebra and its Applications*, 284, 193-228.

    Examples
    --------
    >>> r = second_order_cone([1.0, 0.0], [[[1.0, 0.0], [0.0, 1.0]]], [[0.0, 0.0]], [([0.0, 0.0], 1.0)])
    >>> [round(v, 7) for v in r["x"]], round(r["objective"], 7)
    ([-1.0, 0.0], -1.0)
    """
    f = _vec(c)
    n = len(f)
    cones = []
    for Ai, bi, dom in zip(A, b, domains):
        e, d = dom
        Am, bv, ev = _mat(Ai), _vec(bi), _vec(e)
        if any(len(r) != n for r in Am) or len(ev) != n or len(bv) != len(Am):
            raise ValueError("each cone needs A_i (k x n), b_i (k), e_i (n)")
        cones.append((Am, bv, ev, float(d)))
    if not (len(cones) == len(A) == len(b) == len(domains)):
        raise ValueError("A, b and domains must have the same length")
    m = len(cones)
    x = [0.0] * n if x0 is None else _vec(x0)
    if _barrier(x, cones) is None:
        # phase I in (x, s): min s with ||A x + b|| <= e'x + d + s
        viol = max(
            math.sqrt(sum((sum(a * t for a, t in zip(r, x)) + bb) ** 2 for r, bb in zip(Am, bv)))
            - (sum(a * t for a, t in zip(ev, x)) + d)
            for Am, bv, ev, d in cones
        )
        s = max(viol, 0.0) + 1.0
        ph = [(Am, bv, ev + [1.0], d) for Am, bv, ev, d in cones]
        ph = [([r + [0.0] for r in Am], bv, ev, d) for Am, bv, ev, d in ph]
        z = x + [s]
        fz = [0.0] * n + [1.0]
        t = 1.0
        while True:
            z = _centering(fz, z, ph, t)
            if z[-1] < -1e-8 or m / t < 1e-12:
                break
            t *= 10.0
        if z[-1] >= 0.0:
            return RichResult(payload={"x": None, "objective": None, "feasible": False, "estimate": None})
        x = z[:n]
    t = 1.0
    while True:
        x = _centering(f, x, cones, t)
        if m / t < eps:
            break
        t *= 10.0
    lhs = [
        math.sqrt(sum((sum(a * v for a, v in zip(r, x)) + bb) ** 2 for r, bb in zip(Am, bv)))
        for Am, bv, _e, _d in cones
    ]
    rhs = [sum(a * v for a, v in zip(ev, x)) + d for _A, _b, ev, d in cones]
    obj = sum(a * v for a, v in zip(f, x))
    return RichResult(
        payload={
            "x": x,
            "objective": obj,
            "lhs": lhs,
            "rhs": rhs,
            "slack": [r - lh for r, lh in zip(rhs, lhs)],
            "feasible": True,
            "estimate": obj,
            "method": "Barrier method with generalised-logarithm cone barriers (Boyd and Vandenberghe 2004)",
        }
    )


socpts = second_order_cone


def cheatsheet():
    return "socpts: barrier-method SOCP, min c'x s.t. ||A_i x + b_i|| <= e_i'x + d_i"
