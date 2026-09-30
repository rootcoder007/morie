# morie.fn -- function file (rootcoder007/morie)
"""Research P4: predictive-policing feedback loops (two-region model).

Python twin of ``R/feedback_loop.R`` and of its C++ urn kernel
(``src/morie_feedback_loop.cpp``). A department keeps discovered-crime
counts (cA, cB), sends its patrol to region A with share x = cA/(cA+cB) and
discovers crime only where the patrol is (Ensign, Friedler, Neville,
Scheidegger & Venkatasubramanian 2018). Machine-checked in
``research/lean/P4Feedback.lean``, ``P4Limit.lean``, ``P4Mitigation.lean``
and ``P4Urn.lean``:

- ``Research.P4.naive_step_drift`` / ``naive_share_increasing`` / ``naiveShare_tendsto_one``
- ``Research.P4.corrected_share_tendsto``: the corrected update converges to lamA/(lamA+lamB)
- ``Research.P4.naiveShare_rate_bound``: the share moves at most like a harmonic sum
- ``Research.P4.rho_cap``: public reports cap the naive share at max(x0, lamA/(lamA + rho lamB))
- ``Research.P4.Urn.urn_step_martingale`` / ``polya_uniform`` / ``polya_no_concentration``
"""

from __future__ import annotations

import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "feedback_loop_meanfield",
    "feedback_loop_limit",
    "feedback_loop_urn_law",
    "feedback_loop_sim",
    "feedback_loop_bound",
]


def _check(lam_a, lam_b, c_a0, c_b0, rho):
    for nm, v in (("lam_a", lam_a), ("lam_b", lam_b), ("c_a0", c_a0), ("c_b0", c_b0), ("rho", rho)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v):
            raise ValueError(f"{nm} must be a single non-missing number")
    if lam_a <= 0 or lam_b <= 0:
        raise ValueError("lam_a and lam_b must be positive")
    if c_a0 <= 0 or c_b0 <= 0:
        raise ValueError("c_a0 and c_b0 must be positive")
    if rho < 0 or rho > 1:
        raise ValueError("rho must lie in [0, 1]")


def _update(update):
    if update not in ("naive", "corrected"):
        raise ValueError("update must be 'naive' or 'corrected'")
    return update


def _steps(n_steps):
    n = int(n_steps)
    if n < 0:
        raise ValueError("n_steps must be a non-negative integer")
    return n


def feedback_loop_limit(lam_a, lam_b, c_a0, c_b0, update="naive", rho=0.0):
    """Proved limit of the two-region feedback loop.

    Naive update: the higher-rate region absorbs the whole patrol (share 1
    or 0), equal rates leave the mean-field share where it started.
    Corrected update: the true-rate proportion ``lam_a/(lam_a+lam_b)``.
    With ``rho > 0`` under the naive update no limit is proved, but the
    share never exceeds the cap ``max(x0, lam_a/(lam_a + rho lam_b))``.

    Parameters
    ----------
    lam_a, lam_b : float
        True crime rates per patrol visit; positive.
    c_a0, c_b0 : float
        Initial discovered counts; positive.
    update : {"naive", "corrected"}
    rho : float
        Share of crimes reaching the record through public reports, in [0, 1].

    Returns
    -------
    RichResult
        ``share_a`` (limit or ``nan``), ``theorem``, ``note`` and, for
        ``rho > 0`` naive, ``cap`` and ``cap_theorem``; for equal rates also
        ``stochastic_theorem``.

    Examples
    --------
    >>> feedback_loop_limit(0.3, 0.2, 10, 10).share_a
    1.0
    >>> round(feedback_loop_limit(0.3, 0.2, 10, 10, "corrected").share_a, 12)
    0.6
    >>> round(feedback_loop_limit(0.3, 0.2, 10, 10, rho=0.5).cap, 12)
    0.75
    """
    _update(update)
    _check(lam_a, lam_b, c_a0, c_b0, rho)
    if rho > 0:
        if update == "corrected":
            out = {
                "share_a": lam_a / (lam_a + lam_b),
                "theorem": "Research.P4.corrected_share_tendsto",
                "note": "corrected update: reports do not change the increments",
            }
        else:
            out = {
                "share_a": math.nan,
                "theorem": None,
                "note": "no limit theorem for rho > 0; the cap is proved",
                "cap": max(c_a0 / (c_a0 + c_b0), lam_a / (lam_a + rho * lam_b)),
                "cap_theorem": "Research.P4.rho_cap",
            }
    elif update == "corrected":
        out = {
            "share_a": lam_a / (lam_a + lam_b),
            "theorem": "Research.P4.corrected_share_tendsto",
            "note": "limit is the true-rate proportion, for every start",
        }
    elif lam_a > lam_b:
        out = {
            "share_a": 1.0,
            "theorem": "Research.P4.naiveShare_tendsto_one",
            "note": "runaway: the higher-rate region absorbs the whole patrol",
        }
    elif lam_a < lam_b:
        out = {
            "share_a": 0.0,
            "theorem": "Research.P4.naiveShare_tendsto_one",
            "note": "runaway (regions swapped): the higher-rate region absorbs the whole patrol",
        }
    else:
        out = {
            "share_a": c_a0 / (c_a0 + c_b0),
            "theorem": "Research.P4.naive_step_drift",
            "note": (
                "equal rates: the mean-field drift is zero and the share never moves; "
                "the stochastic urn instead converges to a random limit "
                "(Research.P4.Urn.polya_uniform, see feedback_loop_urn_law)"
            ),
            "stochastic_theorem": "Research.P4.Urn.polya_no_concentration",
        }
    return RichResult(title="Proved limit of the feedback loop", payload=out)


def feedback_loop_meanfield(lam_a, lam_b, c_a0, c_b0, n_steps=100, update="naive", rho=0.0):
    """Mean-field feedback-loop recursion for two regions.

    Iterates the expected discovered-crime counts under a patrol allocation
    proportional to those counts. Naive: increments ``lam_a x`` and
    ``lam_b (1 - x)``; corrected: ``lam_a`` and ``lam_b``. A share ``rho``
    of crimes arrives through public reports split by the true rates.

    Parameters
    ----------
    lam_a, lam_b, c_a0, c_b0, update, rho
        As in :func:`feedback_loop_limit`.
    n_steps : int
        Number of steps to iterate.

    Returns
    -------
    dict
        Columns ``step``, ``share_a``, ``c_a``, ``c_b`` and ``limit``
        (the :func:`feedback_loop_limit` result).

    Examples
    --------
    >>> mf = feedback_loop_meanfield(0.3, 0.2, 10, 10, n_steps=200)
    >>> round(mf["share_a"][-1], 12)
    0.621419442809
    >>> cf = feedback_loop_meanfield(0.3, 0.2, 10, 10, n_steps=200, update="corrected")
    >>> round(cf["share_a"][-1], 12)
    0.583333333333
    """
    _update(update)
    _check(lam_a, lam_b, c_a0, c_b0, rho)
    n = _steps(n_steps)
    c_a = [float(c_a0)]
    c_b = [float(c_b0)]
    p_report = lam_a / (lam_a + lam_b)
    for t in range(n):
        x = c_a[t] / (c_a[t] + c_b[t])
        if update == "naive":
            inc_a = (1 - rho) * lam_a * x
            inc_b = (1 - rho) * lam_b * (1 - x)
        else:
            inc_a = (1 - rho) * lam_a
            inc_b = (1 - rho) * lam_b
        inc_a = inc_a + rho * (lam_a + lam_b) * p_report
        inc_b = inc_b + rho * (lam_a + lam_b) * (1 - p_report)
        c_a.append(c_a[t] + inc_a)
        c_b.append(c_b[t] + inc_b)
    return {
        "step": list(range(n + 1)),
        "share_a": [a / (a + b) for a, b in zip(c_a, c_b)],
        "c_a": c_a,
        "c_b": c_b,
        "limit": feedback_loop_limit(lam_a, lam_b, c_a0, c_b0, update, rho),
    }


def feedback_loop_urn_law(n_steps):
    """Exact law of the stochastic two-region urn with equal rates.

    One discovery per step, credited to A with probability equal to A's
    current share. Started from one count each, the number ``j`` of
    discoveries credited to A after ``n`` draws is uniform on ``0..n``, so
    the share ``(1 + j)/(n + 2)`` is uniform on its grid and lies in
    [1/4, 3/4] with probability at least 1/4 at every horizon ``n >= 2``.

    Parameters
    ----------
    n_steps : int
        Number of draws (non-negative).

    Returns
    -------
    dict
        Columns ``j``, ``share_a``, ``prob`` and ``prob_middle``, ``theorems``.

    Examples
    --------
    >>> law = feedback_loop_urn_law(10)
    >>> law["prob"][0] == 1 / 11, round(law["prob_middle"], 12)
    (True, 0.636363636364)
    """
    n = _steps(n_steps)
    j = list(range(n + 1))
    share = [(1 + v) / (n + 2) for v in j]
    prob = [1 / (n + 1)] * (n + 1)
    return {
        "j": j,
        "share_a": share,
        "prob": prob,
        "prob_middle": math.fsum(p for p, s in zip(prob, share) if 0.25 <= s <= 0.75),
        "theorems": [
            "Research.P4.Urn.urn_step_martingale",
            "Research.P4.Urn.polya_uniform",
            "Research.P4.Urn.polya_no_concentration",
        ],
    }


def _urn_path(lam_a, lam_b, c_a0, c_b0, n_steps, corrected, rho, u):
    """Pure-Python twin of morie_feedback_urn_cpp: consumes 3 uniforms per step
    when rho > 0 (report?, location or visit, discovery) and 2 otherwise."""
    k = 3 if rho > 0 else 2
    ca, cb = float(c_a0), float(c_b0)
    p_rep = lam_a / (lam_a + lam_b)
    share = [ca / (ca + cb)]
    for t in range(n_steps):
        x = ca / (ca + cb)
        base = k * t
        if rho > 0 and u[base] < rho:
            if u[base + 1] < p_rep:
                ca += 1.0
            else:
                cb += 1.0
        else:
            j = base + (1 if rho > 0 else 0)
            visit_a = u[j] < x
            if u[j + 1] < (lam_a if visit_a else lam_b):
                w = 1.0 / (x if visit_a else 1.0 - x) if corrected else 1.0
                if visit_a:
                    ca += w
                else:
                    cb += w
        share.append(ca / (ca + cb))
    return share


def feedback_loop_sim(lam_a, lam_b, c_a0, c_b0, n_steps=1000, update="naive", rho=0.0, n_sims=100, seed=0):
    """Stochastic urn simulation of the two-region feedback loop.

    Each step the patrol visits A with probability equal to the current
    share and discovers a crime there with probability equal to that
    region's true rate; the naive update adds one, the corrected update adds
    one over the presence that produced it. With ``rho > 0`` a step is a
    public report with probability ``rho``, located by the true rates.
    Uniforms come from the Philox stream: run ``i`` (0-based) draws
    ``k * n_steps`` uniforms from ``seed`` on stream ``i`` (``k = 3`` when
    ``rho > 0``, else 2), identical to the R arm.

    Parameters
    ----------
    lam_a, lam_b : float
        Per-visit discovery probabilities in (0, 1].
    c_a0, c_b0, update, rho
        As in :func:`feedback_loop_limit`.
    n_steps : int
        Steps per run.
    n_sims : int
        Number of independent runs.
    seed : int
        Philox seed.

    Returns
    -------
    RichResult
        ``share_a`` (``n_sims`` share paths of length ``n_steps + 1``),
        ``final`` and ``limit``.

    Examples
    --------
    >>> s = feedback_loop_sim(0.3, 0.2, 10, 10, n_steps=50, n_sims=3, seed=1)
    >>> [round(v, 12) for v in s.final]
    [0.628571428571, 0.51724137931, 0.529411764706]
    """
    _update(update)
    _check(lam_a, lam_b, c_a0, c_b0, rho)
    if not (lam_a <= 1 and lam_b <= 1):
        raise ValueError("the urn simulator needs per-visit discovery probabilities: lam_a, lam_b <= 1")
    n = _steps(n_steps)
    n_sims = int(n_sims)
    if n_sims < 1:
        raise ValueError("n_sims must be a positive integer")
    k = 3 if rho > 0 else 2
    paths = []
    for i in range(n_sims):
        u = random_uniform(k * n, seed=seed, stream=i)
        paths.append(_urn_path(lam_a, lam_b, c_a0, c_b0, n, update == "corrected", rho, u))
    return RichResult(
        title="Stochastic feedback-loop urn",
        payload={
            "share_a": paths,
            "final": [p[-1] for p in paths],
            "limit": feedback_loop_limit(lam_a, lam_b, c_a0, c_b0, update, rho),
        },
    )


def feedback_loop_bound(lam_a, lam_b, c_a0, c_b0, n_steps=1000):
    """Proved upper bound on how far the naive loop can move in N steps.

    After ``N`` steps the share to A has risen by at most
    ``(lam_a - lam_b)/4 * sum_{k<N} 1/(c0 + k min(lam_a, lam_b))``, a
    harmonic sum that grows like ``log N``.

    Parameters
    ----------
    lam_a, lam_b : float
        True rates with ``lam_a > lam_b``.
    c_a0, c_b0 : float
        Initial counts; ``c0 = c_a0 + c_b0``.
    n_steps : int
        Horizon ``N``.

    Returns
    -------
    RichResult
        ``bound``, ``share_a_max`` (capped at 1) and ``theorem``.

    Examples
    --------
    >>> round(feedback_loop_bound(0.3, 0.2, 10, 10, n_steps=20000).bound, 12)
    0.663536045685
    """
    _check(lam_a, lam_b, c_a0, c_b0, 0.0)
    if lam_a <= lam_b:
        raise ValueError("the bound is stated for lam_a > lam_b; swap the regions otherwise")
    n = _steps(n_steps)
    lo = min(lam_a, lam_b)
    c0 = c_a0 + c_b0
    bound = (lam_a - lam_b) / 4 * math.fsum(1 / (c0 + k * lo) for k in range(n))
    x0 = c_a0 / (c_a0 + c_b0)
    return RichResult(
        title="Rate bound of the naive feedback loop",
        payload={"bound": bound, "share_a_max": min(1.0, x0 + bound), "theorem": "Research.P4.naiveShare_rate_bound"},
    )


def cheatsheet() -> str:
    return (
        "feedback_loop_meanfield(lam_a, lam_b, c_a0, c_b0, n_steps) -> expected-count recursion (Research P4)\n"
        "feedback_loop_limit(...) -> proved limit: 1 naive, lam_a/(lam_a+lam_b) corrected, rho cap\n"
        "feedback_loop_urn_law(n) -> exact uniform law of the equal-rate Polya urn\n"
        "feedback_loop_sim(..., n_sims, seed) -> Philox urn simulation (same stream as R)\n"
        "feedback_loop_bound(lam_a, lam_b, c_a0, c_b0, n_steps) -> harmonic bound on the share's rise"
    )
