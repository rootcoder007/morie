# morie.fn -- function file (rootcoder007/morie)
"""Combat modelling (analytic operations research): Lanchester square, linear, mixed and logarithmic attrition
laws with reinforcements, closed-form square-law outcomes, attrition-coefficient estimation from battle data,
Hughes' salvo model and Colonel Blotto allocation games."""

from __future__ import annotations

import itertools
import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "lanchester_square_outcome",
    "lanchester_simulate",
    "lanchester_fit",
    "salvo_exchange",
    "blotto_payoff",
    "blotto_best_response",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def lanchester_square_outcome(x0: float, y0: float, a: float, b: float) -> RichResult:
    r"""Closed-form Lanchester (1916) square-law battle with aimed fire and no reinforcements.

    ``dx/dt = -a y``, ``dy/dt = -b x`` (``a`` = Y's per-unit kill rate against
    X). The state equation ``b (x0^2 - x^2) = a (y0^2 - y^2)`` decides the
    winner: X wins iff ``b x0^2 > a y0^2``, with ``sqrt(x0^2 - (a/b) y0^2)``
    survivors, at time ``t* = atanh(y0 sqrt(a/b) / x0) / sqrt(a b)`` (for Y
    winning, swap roles); ``x(t) = x0 cosh(g t) - sqrt(a/b) y0 sinh(g t)``,
    ``g = sqrt(a b)``.

    References
    ----------
    Lanchester, F. W. (1916). *Aircraft in Warfare: The Dawn of the Fourth
    Arm*. Constable, London.
    Taylor, J. G. (1983). *Lanchester Models of Warfare*, Vols I-II.
    Operations Research Society of America.

    Examples
    --------
    >>> r = lanchester_square_outcome(100.0, 60.0, 0.01, 0.01)
    >>> r.winner, round(r.survivors, 6), round(r.duration, 6)
    ('x', 80.0, 69.314718)
    """
    g = math.sqrt(a * b)
    fx, fy = b * x0 * x0, a * y0 * y0
    if fx > fy:
        w, surv = "x", math.sqrt(x0 * x0 - (a / b) * y0 * y0)
        t = math.atanh(y0 * math.sqrt(a / b) / x0) / g
    elif fy > fx:
        w, surv = "y", math.sqrt(y0 * y0 - (b / a) * x0 * x0)
        t = math.atanh(x0 * math.sqrt(b / a) / y0) / g
    else:
        w, surv, t = "draw", 0.0, math.inf
    return RichResult(
        payload={"winner": w, "survivors": surv, "duration": t, "fighting_strength_x": fx, "fighting_strength_y": fy}
    )


def _deriv(law, x, y, a, b, p, q):
    if law == "square":
        return -a * y + p, -b * x + q
    if law == "linear":
        return -a * x * y + p, -b * x * y + q
    if law == "mixed":  # x guerrilla (area fire against it), y conventional (aimed fire against it)
        return -a * x * y + p, -b * x + q
    if law == "logarithmic":
        return -a * x + p, -b * y + q
    raise ValueError("law must be square, linear, mixed or logarithmic")


def lanchester_simulate(
    x0: float,
    y0: float,
    a: float,
    b: float,
    *,
    law: str = "square",
    t_end: float = 100.0,
    dt: float = 0.01,
    reinforce_x: float = 0.0,
    reinforce_y: float = 0.0,
) -> RichResult:
    r"""Integrate a Lanchester attrition law by fourth-order Runge-Kutta with constant reinforcement rates.

    Laws (Taylor 1983): ``square`` ``x' = -a y + P``, ``y' = -b x + Q``
    (aimed fire); ``linear`` ``x' = -a x y + P``, ``y' = -b x y + Q`` (area
    fire / ancient combat); ``mixed`` ``x' = -a x y + P``, ``y' = -b x + Q``
    (Deitchman 1962: guerrilla X under area fire, conventional Y under aimed
    fire); ``logarithmic`` ``x' = -a x + P``, ``y' = -b y + Q`` (Peterson
    1953). Integration stops at ``t_end`` or when a side reaches 0 (the
    crossing time is linearly interpolated within the last step and both
    forces are evaluated there).

    References
    ----------
    Deitchman, S. J. (1962). A Lanchester model of guerrilla warfare.
    *Operations Research*, 10(6), 818-827.
    Taylor, J. G. (1983). *Lanchester Models of Warfare*. ORSA.

    Examples
    --------
    >>> r = lanchester_simulate(100.0, 60.0, 0.01, 0.01, t_end=200.0)
    >>> r.ended_by, round(r.x[-1], 4), round(r.t[-1], 3)
    ('y annihilated', 80.0, 69.315)
    """
    x, y, t = float(x0), float(y0), 0.0
    T, X, Y = [t], [x], [y]
    ended = "time"
    n = int(round(t_end / dt))
    for _ in range(n):
        k1 = _deriv(law, x, y, a, b, reinforce_x, reinforce_y)
        k2 = _deriv(law, x + dt / 2 * k1[0], y + dt / 2 * k1[1], a, b, reinforce_x, reinforce_y)
        k3 = _deriv(law, x + dt / 2 * k2[0], y + dt / 2 * k2[1], a, b, reinforce_x, reinforce_y)
        k4 = _deriv(law, x + dt * k3[0], y + dt * k3[1], a, b, reinforce_x, reinforce_y)
        xn = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        yn = y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if xn <= 0 or yn <= 0:
            fx = x / (x - xn) if xn <= 0 else math.inf
            fy = y / (y - yn) if yn <= 0 else math.inf
            f = min(fx, fy)
            t, x, y = t + f * dt, x + f * (xn - x), y + f * (yn - y)
            if fx <= fy:
                x, ended = 0.0, "x annihilated"
            if fy <= fx:
                y, ended = 0.0, "y annihilated" if ended == "time" else "both annihilated"
            T.append(t)
            X.append(x)
            Y.append(y)
            break
        x, y, t = xn, yn, t + dt
        T.append(t)
        X.append(x)
        Y.append(y)
    return RichResult(payload={"t": T, "x": X, "y": Y, "ended_by": ended})


def lanchester_fit(x, y, *, dt: float = 1.0, law: str = "square", reinforce_x=None, reinforce_y=None) -> RichResult:
    r"""Estimate attrition coefficients from battle force histories.

    For a fixed law the losses ``L_x(t) = x(t) - x(t+1) + P(t)`` (adding any
    reinforcements) are regressed through the origin on the attacking term
    (Engel 1954): ``square`` ``a = sum L_x / (dt sum y)``, ``b = sum L_y /
    (dt sum x)``; ``linear`` uses ``x y``; ``logarithmic`` uses the side's own
    strength. ``law="power"`` fits the generalised law ``L_x = a dt y^p x^q``
    by least squares on logarithms (Bracken 1995; Fricker 1998), returning
    ``a``, ``p``, ``q`` for each side (strengths taken at the start of each
    period; periods with no losses are dropped from the log fit). Also
    returned: the fitted losses and ``R^2`` of the loss predictions.

    References
    ----------
    Engel, J. H. (1954). A verification of Lanchester's law. *Journal of the
    Operations Research Society of America*, 2(2), 163-171.
    Fricker, R. D. (1998). Attrition models of the Ardennes campaign.
    *Naval Research Logistics*, 45(1), 1-22.

    Examples
    --------
    >>> r = lanchester_fit([100.0, 94.0, 88.36], [60.0, 50.0, 40.6])
    >>> round(r.a, 6), round(r.b, 6)
    (0.105818, 0.1)
    """
    X, Y = _vec(x), _vec(y)
    n = len(X) - 1
    P = [0.0] * n if reinforce_x is None else _vec(reinforce_x)
    Q = [0.0] * n if reinforce_y is None else _vec(reinforce_y)
    Lx = [X[t] - X[t + 1] + P[t] for t in range(n)]
    Ly = [Y[t] - Y[t + 1] + Q[t] for t in range(n)]

    def r2(obs, fit):
        m = ssum(obs) / len(obs)
        tss = ssum((v - m) ** 2 for v in obs)
        return 1 - ssum((o - f) ** 2 for o, f in zip(obs, fit)) / tss if tss > 0 else float("nan")

    if law == "power":
        out = {}
        for side, L, own, opp in (("x", Lx, X, Y), ("y", Ly, Y, X)):
            rows = [(math.log(opp[t]), math.log(own[t]), math.log(L[t] / dt)) for t in range(n) if L[t] > 0]
            k = len(rows)
            A = [[1.0, r[0], r[1]] for r in rows]
            z = [r[2] for r in rows]
            M = [[ssum(A[i][u] * A[i][v] for i in range(k)) for v in range(3)] for u in range(3)]
            c = [ssum(A[i][u] * z[i] for i in range(k)) for u in range(3)]
            from ._qpcore import solve

            coef = [float(v) for v in solve(M, c)]
            out[side] = {"a": math.exp(coef[0]), "p": coef[1], "q": coef[2]}
        fx = [out["x"]["a"] * dt * Y[t] ** out["x"]["p"] * X[t] ** out["x"]["q"] for t in range(n)]
        fy = [out["y"]["a"] * dt * X[t] ** out["y"]["p"] * Y[t] ** out["y"]["q"] for t in range(n)]
        return RichResult(
            payload={
                "x": out["x"],
                "y": out["y"],
                "fitted_x": fx,
                "fitted_y": fy,
                "r2_x": r2(Lx, fx),
                "r2_y": r2(Ly, fy),
            }
        )
    if law == "square":
        gx, gy = Y[:n], X[:n]
    elif law == "linear":
        gx = gy = [X[t] * Y[t] for t in range(n)]
    elif law == "logarithmic":
        gx, gy = X[:n], Y[:n]
    else:
        raise ValueError("law must be square, linear, logarithmic or power")
    a = ssum(Lx) / (dt * ssum(gx))
    b = ssum(Ly) / (dt * ssum(gy))
    fx = [a * dt * g for g in gx]
    fy = [b * dt * g for g in gy]
    return RichResult(payload={"a": a, "b": b, "fitted_x": fx, "fitted_y": fy, "r2_x": r2(Lx, fx), "r2_y": r2(Ly, fy)})


def salvo_exchange(
    A: float,
    B: float,
    *,
    alpha: float,
    beta: float,
    a1: float,
    b1: float,
    a3: float = 0.0,
    b3: float = 0.0,
    salvos: int = 1,
) -> RichResult:
    r"""Hughes (1995) salvo model of missile combat between fleets of ``A`` and ``B`` units.

    Per salvo ``Delta B = (alpha A - b3 B) / b1`` and ``Delta A = (beta B - a3
    A) / a1``, each clipped to ``[0, current force]`` (``alpha``, ``beta``
    offensive power per unit, ``a3``, ``b3`` defensive power, ``a1``, ``b1``
    staying power in hits); fires are simultaneous. Returns the force after
    each salvo and the fractional exchange ratio ``(Delta B / B) / (Delta A /
    A)`` of the first salvo.

    References
    ----------
    Hughes, W. P. (1995). A salvo model of warships in missile combat used to
    evaluate their staying power. *Naval Research Logistics*, 42(2),
    267-289.

    Examples
    --------
    >>> r = salvo_exchange(10.0, 8.0, alpha=2.0, beta=2.0, a1=2.0, b1=2.0, a3=1.0, b3=1.0)
    >>> r.A, r.B
    ([10.0, 7.0], [8.0, 2.0])
    """
    As, Bs = [float(A)], [float(B)]
    fer = float("nan")
    for s in range(int(salvos)):
        a_, b_ = As[-1], Bs[-1]
        dB = min(b_, max(0.0, (alpha * a_ - b3 * b_) / b1))
        dA = min(a_, max(0.0, (beta * b_ - a3 * a_) / a1))
        if s == 0 and dA > 0 and a_ > 0 and b_ > 0:
            fer = (dB / b_) / (dA / a_)
        As.append(a_ - dA)
        Bs.append(b_ - dB)
    return RichResult(payload={"A": As, "B": Bs, "fractional_exchange_ratio": fer})


def blotto_payoff(alloc_a, alloc_b, values=None) -> RichResult:
    r"""Colonel Blotto payoff: each battlefield goes to the larger allocation (ties split), weighted by ``values``.

    Returns A's value won, B's value won and the zero-sum payoff ``A - B``
    (Borel 1921; Roberson 2006).

    References
    ----------
    Roberson, B. (2006). The Colonel Blotto game. *Economic Theory*, 29(1), 1-24.

    Examples
    --------
    >>> blotto_payoff([3, 3, 0], [2, 2, 2]).payoff
    1.0
    """
    a, b = _vec(alloc_a), _vec(alloc_b)
    v = [1.0] * len(a) if values is None else _vec(values)
    wa = ssum(vi if x > y else vi / 2 if x == y else 0.0 for x, y, vi in zip(a, b, v))
    wb = ssum(vi if y > x else vi / 2 if x == y else 0.0 for x, y, vi in zip(a, b, v))
    return RichResult(payload={"value_a": wa, "value_b": wb, "payoff": wa - wb})


def _compositions(total, k):
    for cuts in itertools.combinations(range(total + k - 1), k - 1):
        prev, out = -1, []
        for c in cuts:
            out.append(c - prev - 1)
            prev = c
        out.append(total + k - 1 - prev - 1)
        yield out


def blotto_best_response(opponent, troops: int, values=None) -> RichResult:
    r"""Exhaustive best response to a known (mixed) opponent strategy in a discrete Colonel Blotto game.

    ``opponent`` is a list of ``(probability, allocation)`` pairs (or one
    allocation). All integer allocations of ``troops`` over the battlefields
    are enumerated and the one maximising expected :func:`blotto_payoff` is
    returned (first in lexicographic enumeration order on ties).

    Examples
    --------
    >>> r = blotto_best_response([2, 2, 2], 6)
    >>> r.allocation, r.expected_payoff
    ([0, 3, 3], 1.0)
    """
    opp = opponent
    if opp and not isinstance(opp[0], (list, tuple)):
        opp = [(1.0, opp)]
    elif opp and len(opp[0]) == 2 and isinstance(opp[0][1], (list, tuple)):
        opp = [(float(p), s) for p, s in opp]
    k = len(opp[0][1])
    best, arg = -math.inf, None
    for alloc in _compositions(int(troops), k):
        e = ssum(p * blotto_payoff(alloc, s, values)["payoff"] for p, s in opp)
        if e > best + 1e-12:
            best, arg = e, alloc
    return RichResult(payload={"allocation": arg, "expected_payoff": best})


def cheatsheet() -> str:
    return "lanchester_square_outcome / lanchester_simulate / lanchester_fit / salvo_exchange / blotto_payoff."
