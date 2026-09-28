# morie.fn -- function file (rootcoder007/morie)
"""Police and emergency-service staffing: calls-for-service demand forecasting (Poisson harmonic regression by
hour, weekday and season), Erlang-C staffing per period (SIPP), square-root staffing, patrol-car allocation
requirements, exact cyclic shift scheduling, shift-relief factors and workload-based staffing."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = [
    "call_demand_model",
    "erlang_c",
    "staffing_requirement",
    "patrol_car_requirement",
    "shift_schedule",
    "relief_factor",
    "workload_staffing",
]

_YEAR = 365.25 * 24


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _design(t, daily, weekly, annual, trend):
    row = [1.0]
    for k in range(1, daily + 1):
        row += [math.sin(2 * math.pi * k * t / 24), math.cos(2 * math.pi * k * t / 24)]
    for k in range(1, weekly + 1):
        row += [math.sin(2 * math.pi * k * t / 168), math.cos(2 * math.pi * k * t / 168)]
    for k in range(1, annual + 1):
        row += [math.sin(2 * math.pi * k * t / _YEAR), math.cos(2 * math.pi * k * t / _YEAR)]
    if trend:
        row.append(t / _YEAR)
    return row


def call_demand_model(
    counts,
    hours,
    *,
    daily: int = 3,
    weekly: int = 2,
    annual: int = 2,
    trend: bool = True,
    new_hours=None,
    maxit: int = 100,
    epsilon: float = 1e-10,
) -> RichResult:
    r"""Poisson log-linear model of calls for service with daily, weekly and annual harmonics and a linear trend.

    ``log E[N_t] = b0 + sum_k [a_k sin(2 pi k t/24) + c_k cos(2 pi k t/24)] +
    (weekly terms, period 168 h) + (annual terms, period 8766 h) + d t/8766``
    with ``t`` the hour index (hour 0 = the start of a Monday, say). Fitted
    by iteratively reweighted least squares from the ``glm`` start ``mu = y +
    0.1``, stopping when the relative deviance change is below ``epsilon``
    (the ``stats::glm`` rule). Forecasts ``exp(x' b)`` for ``new_hours`` give
    expected calls per hour for any future day, weekday and season, with
    delta-method standard errors of the log mean.

    References
    ----------
    Wilson, J. M. and Weiss, A. (2012). *A Performance-Based Approach to
    Police Staffing and Allocation*. U.S. Department of Justice, Office of
    Community Oriented Policing Services.
    McCullagh, P. and Nelder, J. A. (1989). *Generalized Linear Models*,
    2nd ed. Chapman and Hall.

    Examples
    --------
    >>> t = list(range(48))
    >>> y = [round(5 * math.exp(0.4 * math.sin(2 * math.pi * h / 24))) for h in t]
    >>> r = call_demand_model(y, t, daily=1, weekly=0, annual=0, trend=False)
    >>> round(r.coefficients[1], 4)
    0.3828
    """
    y, T = _vec(counts), _vec(hours)
    X = [_design(t, daily, weekly, annual, trend) for t in T]
    p = len(X[0])
    mu = [v + 0.1 for v in y]
    eta = [math.log(m) for m in mu]
    dev_old = math.inf
    beta = [0.0] * p
    it = 0
    while it < maxit:
        it += 1
        z = [e + (v - m) / m for e, v, m in zip(eta, y, mu)]
        XtWX = [[ssum(X[i][a] * mu[i] * X[i][b] for i in range(len(y))) for b in range(p)] for a in range(p)]
        XtWz = [ssum(X[i][a] * mu[i] * z[i] for i in range(len(y))) for a in range(p)]
        beta = [float(v) for v in solve(XtWX, XtWz)]
        eta = [ssum(b * x for b, x in zip(beta, r)) for r in X]
        mu = [math.exp(e) for e in eta]
        dev = 2 * ssum((v * math.log(v / m) if v > 0 else 0.0) - (v - m) for v, m in zip(y, mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < epsilon:
            break
        dev_old = dev
    XtWX = [[ssum(X[i][a] * mu[i] * X[i][b] for i in range(len(y))) for b in range(p)] for a in range(p)]
    cov = [[float(v) for v in r] for r in np.linalg.inv(np.asarray(XtWX, dtype=float)).tolist()]
    out = {
        "coefficients": beta,
        "fitted": mu,
        "deviance": dev,
        "iterations": it,
        "se": [math.sqrt(cov[k][k]) for k in range(p)],
    }
    if new_hours is not None:
        Xn = [_design(t, daily, weekly, annual, trend) for t in _vec(new_hours)]
        out["forecast"] = [math.exp(ssum(b * x for b, x in zip(beta, r))) for r in Xn]
        out["forecast_log_se"] = [
            math.sqrt(ssum(r[a] * cov[a][b] * r[b] for a in range(p) for b in range(p))) for r in Xn
        ]
    return RichResult(payload=out)


def erlang_c(c: int, a: float) -> float:
    r"""Erlang C probability that a call waits in an ``M/M/c`` queue with offered load ``a = lambda h`` (``nan`` if ``a >= c``).

    Computed from the Erlang B recursion ``B_k = a B_{k-1} / (k + a B_{k-1})``
    and ``C = c B / (c - a (1 - B))``.

    Examples
    --------
    >>> round(erlang_c(3, 2.0), 6)
    0.444444
    """
    if a >= c:
        return float("nan")
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return c * b / (c - a * (1 - b))


def staffing_requirement(
    arrival_rates,
    handle_time: float,
    *,
    target: str = "wait_prob",
    level: float = 0.2,
    threshold: float = 0.0,
    beta: float | None = None,
) -> RichResult:
    r"""Units needed in each period so an ``M/M/c`` queue meets a service target (stationary independent period by period).

    For each period's arrival rate ``lambda`` (calls per hour) and mean
    handling time ``h`` (hours), offered load ``a = lambda h``; the minimal
    ``c > a`` with ``target``: ``wait_prob`` ``C(c, a) <= level``;
    ``service_level`` ``P(W <= threshold) = 1 - C e^{-(c - a) threshold / h}
    >= level``; ``asa`` mean wait ``C h / (c - a) <= level``. With ``beta``
    the square-root staffing rule ``ceil(a + beta sqrt(a))`` (Halfin and Whitt
    1981) is returned alongside.

    References
    ----------
    Green, L. V., Kolesar, P. J. and Whitt, W. (2007). Coping with
    time-varying demand when setting staffing requirements for a service
    system. *Production and Operations Management*, 16(1), 13-39.
    Halfin, S. and Whitt, W. (1981). Heavy-traffic limits for queues with
    many exponential servers. *Operations Research*, 29(3), 567-588.

    Examples
    --------
    >>> staffing_requirement([1.0, 4.0], 1.0, level=0.2).units
    [3, 7]
    """
    out, perf, sqr = [], [], []
    for lam in _vec(arrival_rates):
        a = lam * handle_time
        c = max(1, math.floor(a) + 1)
        while True:
            C = erlang_c(c, a)
            if target == "wait_prob":
                ok, val = level >= C, C
            elif target == "service_level":
                val = 1 - C * math.exp(-(c - a) * threshold / handle_time)
                ok = val >= level
            elif target == "asa":
                val = C * handle_time / (c - a)
                ok = val <= level
            else:
                raise ValueError("target must be wait_prob, service_level or asa")
            if ok:
                break
            c += 1
        out.append(c)
        perf.append(val)
        if beta is not None:
            sqr.append(math.ceil(a + beta * math.sqrt(a)) if a > 0 else 0)
    res = {"units": out, "performance": perf}
    if beta is not None:
        res["square_root_units"] = sqr
    return RichResult(payload=res)


def patrol_car_requirement(
    arrival_rate: float,
    handle_time: float,
    *,
    area: float,
    response_speed: float,
    target_response: float,
    street_miles: float = 0.0,
    patrol_speed: float = 1.0,
    patrol_frequency: float = 0.0,
    target_wait: float = 0.2,
    metric: str = "euclidean",
) -> RichResult:
    r"""Patrol units required in one period as the largest of three requirements (Chaiken and Dormont 1978, PCAM).

    With offered load ``a = lambda h`` (units busy on calls on average):
    queueing ``min c: C(c, a) <= target_wait``; response time, from the
    Kolesar-Blum law ``c_m sqrt(A / (N - a)) / v <= T`` giving ``N >= a + A
    (c_m / (v T))^2`` (``c_m`` 0.5 Euclidean, ``sqrt(2 pi)/4`` rectilinear);
    and preventive patrol frequency ``(N - a) s / L >= f`` giving ``N >= a + f
    L / s`` (street miles ``L`` passed ``f`` times per hour at patrol speed
    ``s``). Returned: each requirement and their maximum (rounded up).

    References
    ----------
    Chaiken, J. M. and Dormont, P. (1978). A patrol car allocation model:
    background. *Management Science*, 24(12), 1280-1290.
    Kolesar, P. and Blum, E. H. (1973). Square root laws for fire engine
    response distances. *Management Science*, 19(12), 1368-1378.

    Examples
    --------
    >>> r = patrol_car_requirement(2.0, 0.5, area=16.0, response_speed=20.0, target_response=0.1)
    >>> r.queue, r.response, r.units
    (3, 2, 3)
    """
    a = arrival_rate * handle_time
    q = staffing_requirement([arrival_rate], handle_time, target="wait_prob", level=target_wait)["units"][0]
    cm = 0.5 if metric == "euclidean" else math.sqrt(2 * math.pi) / 4
    resp = math.ceil(a + area * (cm / (response_speed * target_response)) ** 2 - 1e-12)
    pat = math.ceil(a + patrol_frequency * street_miles / patrol_speed - 1e-12) if patrol_frequency > 0 else 0
    return RichResult(
        payload={"offered_load": a, "queue": q, "response": resp, "patrol": pat, "units": max(q, resp, pat)}
    )


def _feasible(b, L, T):
    n = len(b)
    # nodes 0..n are cumulative starts S_0..S_n; edges u->v with weight w encode S_v - S_u <= w
    edges = []
    for t in range(1, n + 1):
        edges.append((t, t - 1, 0.0))  # S_{t-1} - S_t <= 0 (x_t >= 0)
    edges.append((0, n, T))
    edges.append((n, 0, -T))
    for h in range(1, n + 1):  # hour h (1-based) is covered by shifts starting in hours h-L+1..h
        lo = h - L
        if lo >= 0:
            edges.append((h, lo, -b[h - 1]))  # S_h - S_lo >= b  ->  S_lo - S_h <= -b
        else:
            edges.append((h, lo + n, T - b[h - 1]))  # S_h + T - S_{lo+n} >= b
    dist = [0.0] * (n + 1)
    for _ in range(n + 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v] - 1e-9:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return [dist[t] - dist[0] for t in range(n + 1)]
    return None


def shift_schedule(requirements, shift_length: int) -> RichResult:
    r"""Fewest officers on cyclic shifts of ``shift_length`` consecutive periods covering every period's requirement.

    Minimise ``sum x_s`` over shift start counts subject to each period being
    covered by at least ``b_t`` officers, with the day (or week) wrapping
    around. The constraint matrix has circular consecutive ones, so for a
    fixed total ``T`` feasibility is a system of difference constraints on
    the cumulative starts ``S_t`` (checked with Bellman-Ford) and the optimum
    is the smallest feasible ``T`` (binary search; Bartholdi, Orlin and
    Ratliff 1980). The integer requirements give integer starts.

    References
    ----------
    Dantzig, G. B. (1954). A comment on Edie's "Traffic delays at toll
    booths". *Journal of the Operations Research Society of America*, 2(3),
    339-341.
    Bartholdi, J. J., Orlin, J. B. and Ratliff, H. D. (1980). Cyclic
    scheduling via integer programs with circular ones. *Operations
    Research*, 28(5), 1074-1085.

    Examples
    --------
    >>> r = shift_schedule([2, 2, 3, 5, 5, 4], 3)
    >>> r.total, r.coverage
    (7, [2, 2, 3, 5, 5, 4])
    """
    b = [int(round(v)) for v in _vec(requirements)]
    n, L = len(b), int(shift_length)
    if not 1 <= L <= n:
        raise ValueError("shift_length must be between 1 and the number of periods")
    lo, hi = max(b), sum(b)  # sum(b) officers (one start per unit of requirement) is always feasible
    while lo < hi:
        mid = (lo + hi) // 2
        if _feasible(b, L, mid) is not None:
            hi = mid
        else:
            lo = mid + 1
    S = _feasible(b, L, lo)
    best = [int(round(S[t] - S[t - 1])) for t in range(1, n + 1)]
    cov = [sum(best[(h - k) % n] for k in range(L)) for h in range(n)]
    return RichResult(payload={"starts": best, "total": sum(best), "coverage": cov})


def relief_factor(days_off: float, days_per_year: float = 365.0) -> float:
    r"""Shift-relief factor ``days_per_year / (days_per_year - days_off)``: officers to assign per post staffed every day.

    ``days_off`` sums regular days off, vacation, holidays, sick, training and
    personal leave (Wilson and Weiss 2012; e.g. 151 days gives 365/214 = 1.7).

    Examples
    --------
    >>> round(relief_factor(151), 4)
    1.7056
    """
    return days_per_year / (days_per_year - days_off)


def workload_staffing(
    calls, handle_minutes, *, units_per_call=1.0, call_share: float = 0.5, shift_hours: float = 8.0, relief: float = 1.0
) -> RichResult:
    r"""Workload-based patrol staffing (Wilson and Weiss 2012).

    Obligated hours per shift ``= sum calls x handling minutes / 60 x units
    per call`` (by call type); officers on duty ``= obligated hours / (call
    share x shift hours)`` where ``call_share`` is the fraction of a shift the
    agency wants spent on calls for service (the rest proactive); officers
    assigned ``= on duty x relief factor``. Inputs may be lists by call type.

    Examples
    --------
    >>> r = workload_staffing([30, 10], [45, 90], units_per_call=[1, 2], call_share=0.5, relief=1.7)
    >>> r.obligated_hours, r.on_duty, round(r.assigned, 4)
    (52.5, 13.125, 22.3125)
    """
    C, H = _vec(calls), _vec(handle_minutes)
    U = [float(units_per_call)] * len(C) if isinstance(units_per_call, (int, float)) else _vec(units_per_call)
    obl = ssum(c * h / 60 * u for c, h, u in zip(C, H, U))
    on = obl / (call_share * shift_hours)
    return RichResult(
        payload={
            "obligated_hours": obl,
            "on_duty": on,
            "assigned": on * relief,
            "on_duty_ceiling": math.ceil(on - 1e-12),
        }
    )


def cheatsheet() -> str:
    return "call_demand_model / staffing_requirement / patrol_car_requirement / shift_schedule / relief_factor -> staffing."
