"""Bargaining over spatial policy: the Nash bargaining solution and Rubinstein's alternating offers.

Nash, J. F. (1950). The bargaining problem. Econometrica 18, 155-162. Rubinstein, A. (1982).
Perfect equilibrium in a bargaining model. Econometrica 50, 97-109.
"""

import math

from ._richresult import RichResult

__all__ = ["spatial_bargaining"]


def spatial_bargaining(ideal1, ideal2, status_quo=None, delta1=None, delta2=None):
    r"""Two players with quadratic losses u_i(x) = -||x - x_i||^2 bargain over a policy x.

    Nash solution: maximise (u_1(x) - d_1)(u_2(x) - d_2) with d_i = u_i(status quo) over
    individually rational x. Pareto optima lie on the segment x(t) = x_1 + t (x_2 - x_1),
    where the product is a log-concave quartic in t, maximised by bisection on its derivative over
    the feasible t-interval (Nash 1950). Without a status quo the disagreement point is each
    player's worst point on the segment (the other's ideal), giving t = 1/2.

    Rubinstein alternating offers (player 1 proposes first, discount factors delta_i):
    player 1's share of the surplus is (1 - delta_2) / (1 - delta_1 delta_2) (Rubinstein 1982);
    along the segment with utility linear in distance the agreement is
    x = x_2 + share_1 (x_1 - x_2).

    Parameters
    ----------
    ideal1, ideal2 : points (or scalars)
    status_quo : point, optional
    delta1, delta2 : float in (0, 1), optional

    Returns
    -------
    RichResult
        Keys: nash (point), nash_t, nash_product, and with discount factors
        rubinstein_share, rubinstein (point).

    References
    ----------
    Nash, J. F. (1950). Econometrica 18, 155-162.
    Rubinstein, A. (1982). Econometrica 50, 97-109.

    Examples
    --------
    >>> [round(v, 9) for v in spatial_bargaining([0.0], [1.0])["nash"]]
    [0.5]
    """
    a = [float(ideal1)] if isinstance(ideal1, (int, float)) else [float(v) for v in ideal1]
    b = [float(ideal2)] if isinstance(ideal2, (int, float)) else [float(v) for v in ideal2]
    d = len(a)

    def u(x, i):
        s = 0.0
        for k in range(d):
            s += (x[k] - (a if i == 0 else b)[k]) ** 2
        return -s

    def pt(t):
        return [a[k] + t * (b[k] - a[k]) for k in range(d)]

    if status_quo is None:
        d1, d2 = u(b, 0), u(a, 1)
    else:
        q = [float(status_quo)] if isinstance(status_quo, (int, float)) else [float(v) for v in status_quo]
        d1, d2 = u(q, 0), u(q, 1)
    L = 0.0
    for k in range(d):
        L += (b[k] - a[k]) ** 2
    # feasible t: u_1(t) = -t^2 L >= d1 and u_2(t) = -(1 - t)^2 L >= d2
    lo = max(0.0, 1 - math.sqrt(-d2 / L)) if L > 0 else 0.0
    hi = min(1.0, math.sqrt(-d1 / L)) if L > 0 else 0.0
    out = {}
    if hi < lo:
        out.update(nash=None, nash_t=None, nash_product=None)
    else:
        f = lambda t: (-(t * t) * L - d1) * (-((1 - t) ** 2) * L - d2)  # noqa: E731

        def fprime(t):
            A, B = -(t * t) * L - d1, -((1 - t) ** 2) * L - d2
            return -2 * t * L * B + 2 * (1 - t) * L * A

        # the product of two positive concave quadratics is log-concave: f' changes sign once
        x0, x1 = lo, hi
        for _ in range(200):
            mid = 0.5 * (x0 + x1)
            if fprime(mid) > 0:
                x0 = mid
            else:
                x1 = mid
        t = 0.5 * (x0 + x1)
        out.update(nash=pt(t), nash_t=t, nash_product=f(t))
    if delta1 is not None and delta2 is not None:
        s1 = (1 - float(delta2)) / (1 - float(delta1) * float(delta2))
        out["rubinstein_share"] = s1
        out["rubinstein"] = [b[k] + s1 * (a[k] - b[k]) for k in range(d)]
    return RichResult(title="Spatial bargaining", summary_lines=[("Nash t", out.get("nash_t"))], payload=out)


def cheatsheet():
    return "svbarg: Nash bargaining solution and Rubinstein alternating offers on a policy segment"
