# morie.fn -- function file (rootcoder007/morie)
"""Maximising a quadratic (weighted Euclidean) spatial utility under equality constraints (Lagrange) and
inequality constraints (KKT active sets), and the Vickrey-Clarke-Groves mechanism."""

from __future__ import annotations

import itertools

from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = ["quadratic_utility_lagrange", "quadratic_utility_constrained", "vcg_mechanism"]


def _eq_qp(xs, W, C, d):
    """Maximise -(x - xs)' W (x - xs) s.t. C x = d; returns (x, lambda) with gradient 2W(xs - x) = C' lambda."""
    p = len(xs)
    Wi = inverse(W)
    if not C:
        return list(xs), []
    m = len(C)
    CWi = [[ssum(C[a][k] * Wi[k][j] for k in range(p)) for j in range(p)] for a in range(m)]
    M = [[ssum(CWi[a][k] * C[b][k] for k in range(p)) for b in range(m)] for a in range(m)]
    r = [ssum(C[a][k] * xs[k] for k in range(p)) - d[a] for a in range(m)]
    Mi = inverse(M)
    mu = [ssum(Mi[a][b] * r[b] for b in range(m)) for a in range(m)]
    x = [xs[i] - ssum(CWi[a][i] * mu[a] for a in range(m)) for i in range(p)]
    return x, [2.0 * v for v in mu]


def _utility(x, xs, W):
    p = len(x)
    return -ssum((x[i] - xs[i]) * W[i][j] * (x[j] - xs[j]) for i in range(p) for j in range(p))


def quadratic_utility_lagrange(ideal, W, C, d) -> RichResult:
    r"""Utility-maximising policy under linear equality constraints, by Lagrange multipliers.

    Maximises ``u(x) = -(x - x*)' W (x - x*)`` (``W`` positive definite
    salience matrix, ``x*`` the ideal point) subject to ``C x = d``:
    ``x = x* - W^{-1} C' (C W^{-1} C')^{-1} (C x* - d)`` with multipliers
    ``lambda = 2 (C W^{-1} C')^{-1} (C x* - d)`` (the marginal utility of
    relaxing each constraint, ``grad u = C' lambda``).

    References
    ----------
    Hinich, M. J. and Munger, M. C. (1997). *Analytical Politics*. Cambridge
    University Press, ch. 3.
    Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press, section 10.1.

    Examples
    --------
    >>> r = quadratic_utility_lagrange([2.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]], [1.0])
    >>> [round(v, 12) for v in r.x], [round(v, 12) for v in r.multipliers]
    ([1.0, 0.0], [2.0])
    """
    xs = [float(v) for v in ideal]
    Wm = [[float(v) for v in row] for row in W]
    Cm = [[float(v) for v in row] for row in C]
    x, lam = _eq_qp(xs, Wm, Cm, [float(v) for v in d])
    return RichResult(payload={"x": x, "multipliers": lam, "utility": _utility(x, xs, Wm)})


def quadratic_utility_constrained(ideal, W, A, b, *, tol: float = 1e-10) -> RichResult:
    r"""Utility-maximising policy on the polyhedron ``A x <= b`` by enumerating KKT active sets.

    For each candidate active set the equality-constrained problem is solved
    in closed form; the solution is accepted when it is primal feasible and
    all multipliers are non-negative (KKT conditions, sufficient because the
    utility is strictly concave). Exact and suited to the few constraints of
    spatial choice problems (enumeration is exponential in their number).

    References
    ----------
    Kuhn, H. W. and Tucker, A. W. (1951). Nonlinear programming. *Proc. Second
    Berkeley Symposium*, 481-492.
    Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd edn. Springer, ch. 16.

    Examples
    --------
    >>> r = quadratic_utility_constrained([2.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], [[1, 0], [0, 1]], [1.0, 3.0])
    >>> [round(v, 12) for v in r.x], r.active
    ([1.0, 2.0], [0])
    """
    xs = [float(v) for v in ideal]
    Wm = [[float(v) for v in row] for row in W]
    Am = [[float(v) for v in row] for row in A]
    bm = [float(v) for v in b]
    m = len(Am)
    for r in range(0, min(m, len(xs)) + 1):
        for S in itertools.combinations(range(m), r):
            try:
                x, lam = _eq_qp(xs, Wm, [Am[i] for i in S], [bm[i] for i in S])
            except ZeroDivisionError:
                continue
            feas = all(ssum(Am[i][k] * x[k] for k in range(len(x))) <= bm[i] + tol for i in range(m))
            if feas and all(v >= -tol for v in lam):
                mult = [0.0] * m
                for i, v in zip(S, lam):
                    mult[i] = v
                return RichResult(
                    payload={"x": x, "multipliers": mult, "active": list(S), "utility": _utility(x, xs, Wm)}
                )
    raise ValueError("no feasible KKT point: the constraints may be inconsistent")


def vcg_mechanism(values) -> RichResult:
    r"""Vickrey-Clarke-Groves mechanism with Clarke pivot payments.

    ``values[i][k]`` is player ``i``'s value for alternative ``k``. The
    efficient alternative maximises total reported value (ties to the lower
    index); player ``i`` pays the externality it imposes,
    ``max_k sum_{j != i} v_jk - sum_{j != i} v_{j k*}``. Truthful reporting is
    a dominant strategy (incentive compatible), and payments are non-negative.

    References
    ----------
    Vickrey, W. (1961). Counterspeculation, auctions, and competitive sealed
    tenders. *Journal of Finance*, 16, 8-37.
    Clarke, E. H. (1971). Multipart pricing of public goods. *Public Choice*, 11, 17-33.
    Groves, T. (1973). Incentives in teams. *Econometrica*, 41, 617-631.

    Examples
    --------
    >>> r = vcg_mechanism([[5, 0], [0, 3], [0, 4]])
    >>> r.choice, r.payments
    (1, [0.0, 1.0, 2.0])
    """
    V = [[float(v) for v in row] for row in values]
    n, K = len(V), len(V[0])
    tot = [ssum(V[i][k] for i in range(n)) for k in range(K)]
    ch = max(range(K), key=lambda k: (tot[k], -k))
    pay = []
    for i in range(n):
        others = [tot[k] - V[i][k] for k in range(K)]
        pay.append(max(others) - others[ch])
    return RichResult(
        payload={"choice": ch, "payments": pay, "utilities": [V[i][ch] - pay[i] for i in range(n)], "welfare": tot[ch]}
    )


def cheatsheet() -> str:
    return "quadratic_utility_lagrange / quadratic_utility_constrained / vcg_mechanism -> utility maximisation and mechanisms."
