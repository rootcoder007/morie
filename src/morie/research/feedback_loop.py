# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P4: predictive-policing feedback loops (two-region model)
(``research/lean/P4Feedback.lean``, ``P4Limit.lean``, ``P4Rate.lean``, ``P4Mitigation.lean``, ``P4Urn.lean``).

* ``Research.P4.naive_step_drift``: x' - x = x(1-x)(lamA-lamB)/(c + lamA x + lamB(1-x))
* ``Research.P4.naive_share_increasing`` / ``naiveShare_tendsto_one`` / ``corrected_share_tendsto``
* ``Research.P4.naiveShare_rate_bound``: x_N - x_0 <= (lamA-lamB)/4 sum_{k<N} 1/(c0 + k min(lamA, lamB))
* ``Research.P4.rho_cap`` / ``cap_zero`` / ``cap_one``
* ``Research.P4.Urn.urn_step_martingale`` / ``polya_uniform`` / ``polya_no_concentration``

The urn simulator reproduces the R package's C++ kernel step for step on the
shared Philox stream, so both arms give identical paths for a seed.

R parity: ``rmorie`` ``R/feedback_loop.R``, ``src/morie_feedback_loop.cpp``.
"""

from __future__ import annotations

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn._rng import random_uniform

__all__ = [
    "feedback_loop_meanfield",
    "feedback_loop_limit",
    "feedback_loop_urn_law",
    "feedback_loop_sim",
    "feedback_loop_bound",
]


def _check(lam_a, lam_b, c_a0, c_b0, rho):
    for name, v in (("lam_a", lam_a), ("lam_b", lam_b), ("c_a0", c_a0), ("c_b0", c_b0), ("rho", rho)):
        if isinstance(v, bool) or not isinstance(v, int | float) or v != v:
            raise ValueError(f"{name} must be a single non-missing number")
    if lam_a <= 0 or lam_b <= 0:
        raise ValueError("lam_a and lam_b must be positive")
    if c_a0 <= 0 or c_b0 <= 0:
        raise ValueError("c_a0 and c_b0 must be positive")
    if rho < 0 or rho > 1:
        raise ValueError("rho must lie in [0, 1]")


def feedback_loop_meanfield(lam_a, lam_b, c_a0, c_b0, n_steps=100, update="naive", rho=0.0):
    """Mean-field feedback-loop recursion for two regions.

    Returns a frame with ``step, share_a, c_a, c_b``; ``frame.attrs["limit"]``
    holds the proved limit (see :func:`feedback_loop_limit`).

    Examples
    --------
    >>> mf = feedback_loop_meanfield(0.3, 0.2, 7, 3, n_steps=1)
    >>> x = 0.7; drift = x * (1 - x) * 0.1 / (10 + 0.3 * x + 0.2 * (1 - x))
    >>> round(float(mf["share_a"][1] - mf["share_a"][0]) - drift, 15)
    0.0
    """
    if update not in ("naive", "corrected"):
        raise ValueError("update must be 'naive' or 'corrected'")
    _check(lam_a, lam_b, c_a0, c_b0, rho)
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    c_a = [0.0] * (n_steps + 1)
    c_b = [0.0] * (n_steps + 1)
    c_a[0] = float(c_a0)
    c_b[0] = float(c_b0)
    p_report = lam_a / (lam_a + lam_b)
    for t in range(n_steps):
        x = c_a[t] / (c_a[t] + c_b[t])
        if update == "naive":
            inc_a = (1 - rho) * lam_a * x
            inc_b = (1 - rho) * lam_b * (1 - x)
        else:
            inc_a = (1 - rho) * lam_a
            inc_b = (1 - rho) * lam_b
        inc_a += rho * (lam_a + lam_b) * p_report
        inc_b += rho * (lam_a + lam_b) * (1 - p_report)
        c_a[t + 1] = c_a[t] + inc_a
        c_b[t + 1] = c_b[t] + inc_b
    ca = np.array(c_a)
    cb = np.array(c_b)
    out = pd.DataFrame({"step": np.arange(n_steps + 1), "share_a": ca / (ca + cb), "c_a": ca, "c_b": cb})
    out.attrs["limit"] = feedback_loop_limit(lam_a, lam_b, c_a0, c_b0, update, rho)
    return out


def feedback_loop_limit(lam_a, lam_b, c_a0, c_b0, update="naive", rho=0.0) -> dict:
    """Proved limit of the two-region feedback loop, with the theorem that proves it.

    Examples
    --------
    >>> feedback_loop_limit(0.3, 0.2, 10, 10)["share_a"]
    1
    >>> feedback_loop_limit(0.3, 0.2, 10, 10, "corrected")["share_a"]
    0.6
    >>> round(feedback_loop_limit(0.3, 0.2, 10, 10, rho=0.5)["cap"], 12)
    0.75
    """
    if update not in ("naive", "corrected"):
        raise ValueError("update must be 'naive' or 'corrected'")
    _check(lam_a, lam_b, c_a0, c_b0, rho)
    if rho > 0:
        if update == "corrected":
            return {
                "share_a": lam_a / (lam_a + lam_b),
                "theorem": "Research.P4.corrected_share_tendsto",
                "note": "corrected update: reports do not change the increments",
            }
        return {
            "share_a": float("nan"),
            "theorem": None,
            "note": "no limit theorem for rho > 0; the cap is proved",
            "cap": max(c_a0 / (c_a0 + c_b0), lam_a / (lam_a + rho * lam_b)),
            "cap_theorem": "Research.P4.rho_cap",
        }
    if update == "corrected":
        return {
            "share_a": lam_a / (lam_a + lam_b),
            "theorem": "Research.P4.corrected_share_tendsto",
            "note": "limit is the true-rate proportion, for every start",
        }
    if lam_a > lam_b:
        return {
            "share_a": 1,
            "theorem": "Research.P4.naiveShare_tendsto_one",
            "note": "runaway: the higher-rate region absorbs the whole patrol",
        }
    if lam_a < lam_b:
        return {
            "share_a": 0,
            "theorem": "Research.P4.naiveShare_tendsto_one",
            "note": "runaway (regions swapped): the higher-rate region absorbs the whole patrol",
        }
    return {
        "share_a": c_a0 / (c_a0 + c_b0),
        "theorem": "Research.P4.naive_step_drift",
        "note": "equal rates: the mean-field drift is zero and the share never moves; the stochastic urn "
        "instead converges to a random limit (Research.P4.Urn.polya_uniform, see feedback_loop_urn_law)",
        "stochastic_theorem": "Research.P4.Urn.polya_no_concentration",
    }


def feedback_loop_urn_law(n_steps):
    """Exact law of the stochastic two-region urn with equal rates (``polya_uniform``).

    Examples
    --------
    >>> law = feedback_loop_urn_law(10)
    >>> (round(float(law["prob"][0]), 12), law.attrs["prob_middle"] >= 0.25)
    (0.090909090909, True)
    """
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    j = np.arange(n_steps + 1)
    share = (1 + j) / (n_steps + 2)
    prob = np.full(n_steps + 1, 1 / (n_steps + 1))
    out = pd.DataFrame({"j": j, "share_a": share, "prob": prob})
    out.attrs["prob_middle"] = float(np.sum(prob[(share >= 0.25) & (share <= 0.75)]))
    out.attrs["theorems"] = [
        "Research.P4.Urn.urn_step_martingale",
        "Research.P4.Urn.polya_uniform",
        "Research.P4.Urn.polya_no_concentration",
    ]
    return out


def _urn(lam_a, lam_b, c_a0, c_b0, n_steps, corrected, rho, u):
    """The C++ kernel of the R package, step for step."""
    k = 3 if rho > 0 else 2
    share = [0.0] * (n_steps + 1)
    c_a = float(c_a0)
    c_b = float(c_b0)
    p_report = lam_a / (lam_a + lam_b)
    share[0] = c_a / (c_a + c_b)
    for t in range(n_steps):
        x = c_a / (c_a + c_b)
        base = k * t
        if rho > 0 and u[base] < rho:
            if u[base + 1] < p_report:
                c_a += 1.0
            else:
                c_b += 1.0
        else:
            j = base + (1 if rho > 0 else 0)
            visit_a = u[j] < x
            lam = lam_a if visit_a else lam_b
            if u[j + 1] < lam:
                presence = x if visit_a else (1.0 - x)
                w = 1.0 / presence if corrected else 1.0
                if visit_a:
                    c_a += w
                else:
                    c_b += w
        share[t + 1] = c_a / (c_a + c_b)
    return share


def feedback_loop_sim(lam_a, lam_b, c_a0, c_b0, n_steps=1000, update="naive", rho=0.0, n_sims=100, seed=0) -> dict:
    """Stochastic urn simulation of the two-region feedback loop on the shared Philox stream.

    Examples
    --------
    >>> s = feedback_loop_sim(0.3, 0.2, 10, 10, n_steps=50, n_sims=3, seed=1)
    >>> (s["share_a"].shape, s["limit"]["theorem"])
    ((3, 51), 'Research.P4.naiveShare_tendsto_one')
    """
    if update not in ("naive", "corrected"):
        raise ValueError("update must be 'naive' or 'corrected'")
    _check(lam_a, lam_b, c_a0, c_b0, rho)
    if not (lam_a <= 1 and lam_b <= 1):
        raise ValueError("the urn simulator needs per-visit discovery probabilities: lam_a, lam_b <= 1")
    n_steps = int(n_steps)
    n_sims = int(n_sims)
    if n_sims < 1:
        raise ValueError("n_sims must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int | float) or seed != seed or seed < 0:
        raise ValueError("seed must be a single non-negative number")
    k = 3 if rho > 0 else 2
    paths = np.zeros((n_sims, n_steps + 1))
    for i in range(n_sims):
        u = [float(v) for v in random_uniform(k * n_steps, seed=int(seed), stream=i)]
        paths[i, :] = np.array(_urn(lam_a, lam_b, c_a0, c_b0, n_steps, update == "corrected", rho, u))
    return {
        "share_a": paths,
        "final": paths[:, n_steps],
        "limit": feedback_loop_limit(lam_a, lam_b, c_a0, c_b0, update, rho),
    }


def feedback_loop_bound(lam_a, lam_b, c_a0, c_b0, n_steps=1000) -> dict:
    """Proved upper bound on how far the naive loop can move in N steps (``naiveShare_rate_bound``).

    Examples
    --------
    >>> round(feedback_loop_bound(0.3, 0.2, 10, 10, n_steps=4)["bound"], 12)
    0.004926706191
    """
    _check(lam_a, lam_b, c_a0, c_b0, 0.0)
    if lam_a <= lam_b:
        raise ValueError("the bound is stated for lam_a > lam_b; swap the regions otherwise")
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    m = min(lam_a, lam_b)
    bound = (lam_a - lam_b) / 4 * sum(1 / (c_a0 + c_b0 + k * m) for k in range(n_steps))
    x0 = c_a0 / (c_a0 + c_b0)
    return {"bound": bound, "share_a_max": min(1.0, x0 + bound), "theorem": "Research.P4.naiveShare_rate_bound"}
