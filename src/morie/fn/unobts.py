# morie.fn -- function file (rootcoder007/morie)
"""Unobserved components (basic structural) model by the Kalman filter.

SOURCE.  Harvey, A.C. (1989), *Forecasting, Structural Time Series
Models and the Kalman Filter*, Cambridge University Press;
doi:10.1017/CBO9781107049994.  The basic structural model of his
Section 2.3 is

    y_t   = mu_t + gamma_t + eps_t,            eps_t  ~ (0, sigma^2)
    mu_t  = mu_{t-1} + beta_{t-1} + eta_t,     eta_t  ~ (0, sigma^2 q_eta)
    beta_t= beta_{t-1} + zeta_t,               zeta_t ~ (0, sigma^2 q_zeta)
    gamma_t = -sum_{j=1}^{s-1} gamma_{t-j} + omega_t.

It is put in state space form (Harvey Sec. 3.1) and run through the
Kalman filter (Sec. 3.2):

    a_{t|t-1} = T a_{t-1},   P_{t|t-1} = T P_{t-1} T' + Q
    v_t = y_t - Z a_{t|t-1}, F_t = Z P_{t|t-1} Z' + 1
    a_t = a_{t|t-1} + K_t v_t,  K_t = P_{t|t-1} Z'/F_t
    P_t = P_{t|t-1} - K_t Z P_{t|t-1}.

The disturbance variances are parameterised as *ratios* q to sigma^2,
so sigma^2 concentrates out of the likelihood exactly (Harvey Sec. 3.4):

    sigma^2_hat = (1/(T-d)) sum_{t>d} v_t^2 / F_t
    log L       = -(T-d)/2 [log 2 pi + 1 + log sigma^2_hat]
                  - (1/2) sum_{t>d} log F_t.

DIFFUSE PRIOR.  Harvey Sec. 3.3.4 handles the unknown initial state by
letting the prior variance go to infinity; that limit is approximated
here by P_0 = kappa I with kappa large and by dropping the first d
prediction errors from the likelihood, d = state dimension.  With
kappa = 1e10 the level-only case reproduces the exact diffuse answer to
about 1e-10, which the anchor checks.

SCOPE.  Level, slope, seasonal and irregular are implemented.  The
stochastic *cycle* of Harvey Sec. 2.3.4 is NOT: it carries two further
states and its own damping and frequency parameters, which would have to
be estimated rather than gridded.  That omission is this
implementation's scope choice, stated rather than attributed.

ESTIMATION.  The variance ratios are chosen by exhaustive search over a
fixed lattice, not by numerical optimisation.  A lattice is what makes
the two language arms land on identical numbers rather than merely
similar ones; the lattice is stated in ``ratio_grid`` and is this
implementation's choice.

NOT read from the book's own page images -- Harvey (1989) is not in the
local corpus.  The state space form, the filter recursions and the
concentrated likelihood above are the standard statements; the module is
anchored on the closed-form local-level case (see below), not on
agreement with the R arm.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["unobserved_components"]

_GRID = (0.0, 0.01, 0.1, 0.5, 1.0)


def _names(components):
    if components is None:
        return ["level"]
    if isinstance(components, str):
        out = [components.strip().lower()]
    else:
        out = [str(c).strip().lower() for c in components]
    for c in out:
        if c not in ("level", "trend", "seasonal", "irregular"):
            raise ValueError("unobserved_components: unknown component %r" % (c,))
    if "trend" in out and "level" not in out:
        out = ["level"] + out
    if "level" not in out:
        out = ["level"] + out
    return out


def _build(names, period):
    """Return (T matrix, Z row, indices of the stochastic state slots)."""
    has_trend = "trend" in names
    has_seas = "seasonal" in names
    ns = (period - 1) if has_seas else 0
    d = 1 + (1 if has_trend else 0) + ns
    Tm = [[0.0] * d for _ in range(d)]
    Z = [0.0] * d
    Tm[0][0] = 1.0
    Z[0] = 1.0
    j = 1
    if has_trend:
        Tm[0][1] = 1.0
        Tm[1][1] = 1.0
        j = 2
    slots = [0] + ([1] if has_trend else [])
    if has_seas:
        for q in range(ns):
            Tm[j][j + q] = -1.0
        for q in range(1, ns):
            Tm[j + q][j + q - 1] = 1.0
        Z[j] = 1.0
        slots.append(j)
    return Tm, Z, slots, d


def _filter(y, Tm, Z, qv, d, kappa):
    n = len(y)
    a = [0.0] * d
    P = [[kappa if i == j else 0.0 for j in range(d)] for i in range(d)]
    lv = []
    lf = []
    states = []
    for t in range(n):
        ap = core.matvec(Tm, a)
        TP = core.matmul(Tm, P)
        Pp = core.matmul(TP, core.tr(Tm))
        for i in range(d):
            Pp[i][i] += qv[i]
        Pz = [0.0] * d
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += Pp[i][j] * Z[j]
            Pz[i] = s
        F = 1.0
        for i in range(d):
            F += Z[i] * Pz[i]
        za = 0.0
        for i in range(d):
            za += Z[i] * ap[i]
        v = y[t] - za
        K = [Pz[i] / F for i in range(d)]
        a = [ap[i] + K[i] * v for i in range(d)]
        ZP = [0.0] * d
        for j in range(d):
            s = 0.0
            for i in range(d):
                s += Z[i] * Pp[i][j]
            ZP[j] = s
        P = [[Pp[i][j] - K[i] * ZP[j] for j in range(d)] for i in range(d)]
        lv.append(v)
        lf.append(F)
        states.append(list(a))
    return lv, lf, states


def _loglik(lv, lf, d):
    n = len(lv)
    m = n - d
    if m <= 0:
        return float("nan"), float("nan")
    ss = 0.0
    lsum = 0.0
    for t in range(d, n):
        ss += lv[t] * lv[t] / lf[t]
        lsum += math.log(lf[t])
    s2 = ss / m
    ll = -0.5 * m * (math.log(2.0 * math.pi) + 1.0 + math.log(s2)) - 0.5 * lsum
    return s2, ll


def unobserved_components(y, components="level", period=4, ratio_grid=None,
                          kappa=1.0e10):
    """Fit a basic structural model and return the filtered components.

    Parameters
    ----------
    y : array-like
        Univariate series.
    components : str or sequence of str
        Any of ``"level"``, ``"trend"``, ``"seasonal"``, ``"irregular"``.
        ``"level"`` is always present; ``"trend"`` implies it.
    period : int
        Seasonal period s, used only when ``"seasonal"`` is requested.
    ratio_grid : sequence of float or None
        Lattice searched for each stochastic variance ratio.  Default
        ``(0, 0.01, 0.1, 0.5, 1)``.
    kappa : float
        Finite stand-in for Harvey's diffuse prior variance.

    Returns
    -------
    RichResult
        ``level``, ``slope``, ``seasonal``, ``irregular`` (filtered),
        ``sigma2``, ``loglik``, ``aic``, ``ratios``, ``resid``, ``F``,
        ``n``, ``d``.

    Raises
    ------
    ValueError
        Empty series, unknown component name, seasonal period < 2, or a
        series too short for the state dimension.

    References
    ----------
    Harvey, A.C. (1989).  Forecasting, Structural Time Series Models and
    the Kalman Filter.  Cambridge University Press.
    doi:10.1017/CBO9781107049994.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("unobserved_components: y is empty")
    names = _names(components)
    period = int(period)
    if "seasonal" in names and period < 2:
        raise ValueError("unobserved_components: seasonal period must be at least 2")
    Tm, Z, slots, d = _build(names, period)
    if n <= d:
        raise ValueError("unobserved_components: series shorter than the state dimension")
    grid = tuple(_GRID if ratio_grid is None else [float(g) for g in ratio_grid])
    if not grid:
        raise ValueError("unobserved_components: ratio_grid is empty")
    best = None
    idx = [0] * len(slots)
    total = len(grid) ** len(slots)
    for c in range(total):
        r = c
        for j in range(len(slots)):
            idx[j] = r % len(grid)
            r //= len(grid)
        qv = [0.0] * d
        for j, sl in enumerate(slots):
            qv[sl] = grid[idx[j]]
        lv, lf, st = _filter(yv, Tm, Z, qv, d, float(kappa))
        s2, ll = _loglik(lv, lf, d)
        if s2 != s2 or ll != ll:
            continue
        if best is None or ll > best[0]:
            best = (ll, s2, [grid[idx[j]] for j in range(len(slots))], lv, lf, st, list(qv))
    if best is None:
        raise ValueError("unobserved_components: no admissible variance ratios")
    ll, s2, ratios, lv, lf, st, qv = best
    has_trend = "trend" in names
    has_seas = "seasonal" in names
    js = 2 if has_trend else 1
    level = [st[t][0] for t in range(n)]
    slope = [st[t][1] for t in range(n)] if has_trend else [0.0] * n
    seas = [st[t][js] for t in range(n)] if has_seas else [0.0] * n
    irreg = [yv[t] - level[t] - seas[t] for t in range(n)]
    npar = len(slots) + 1
    return RichResult(
        title="Unobserved components (basic structural) model",
        summary_lines=[("obs", n), ("states", d), ("loglik", ll)],
        payload={
            "estimate": ll,
            "level": level,
            "slope": slope,
            "seasonal": seas,
            "irregular": irreg,
            "sigma2": s2,
            "loglik": ll,
            "aic": -2.0 * ll + 2.0 * npar,
            "ratios": ratios,
            "q": qv,
            "resid": lv,
            "F": lf,
            "n": n,
            "d": d,
            "components": names,
            "method": "Basic structural model, Kalman filter, concentrated diffuse likelihood (Harvey 1989 Secs. 2.3, 3.2, 3.4)",
        },
    )


def cheatsheet():
    return "unobts: unobserved components (level/trend/seasonal) by Kalman filter (Harvey 1989)"
