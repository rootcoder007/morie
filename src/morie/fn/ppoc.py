# morie.fn -- slice s03 (rootcoder007/morie)
"""PPO's clipped surrogate objective.

Source consulted (FETCHED, arXiv:1707.06347 via pdftotext): Schulman,
J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017).
Proximal policy optimization algorithms.  Equation (7):

    L^CLIP(theta) = E_t[ min( r_t(theta) A_t,
                              clip(r_t(theta), 1 - eps, 1 + eps) A_t ) ]

with r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t) the
probability ratio, and the paper's remark that "we take the minimum of
the clipped and unclipped objective, so the final objective is a lower
bound ... on the unclipped objective".  Equation (9) adds the value and
entropy terms,

    L^(CLIP+VF+S) = E_t[ L^CLIP - c1 L^VF + c2 S[pi](s_t) ]

which are included when the value targets and the policy entropy are
supplied.  The objective is a quantity to be *maximised*.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["ppo"]


def ppo(env, policy=None, clip_eps=0.2, ratio=None, adv=None, logp_new=None,
        logp_old=None, v_pred=None, v_targ=None, entropy=None,
        c1=0.5, c2=0.01):
    """The clipped surrogate, and optionally the full PPO objective.

    Parameters
    ----------
    env : array-like
        The advantages A_t.  (Positional, for signature stability.)
    policy : array-like, optional
        The probability ratios r_t.  Ignored when ``ratio`` or the two
        log-probability vectors are given.
    clip_eps : float
        The clipping range eps; the paper uses 0.2.
    ratio, adv : array-like, optional
        Explicit ratios and advantages, overriding ``env``/``policy``.
    logp_new, logp_old : array-like, optional
        Log probabilities; the ratio is then exp(new - old).
    v_pred, v_targ : array-like, optional
        Value predictions and targets, for the L^VF term.
    entropy : array-like, optional
        Per-step policy entropy, for the S term.
    c1, c2 : float
        Value and entropy coefficients of equation (9).

    Returns
    -------
    RichResult with payload:
        estimate    : L^CLIP
        l_clip, l_vf, l_entropy, total
        frac_clipped: share of steps where the clip was active
    """
    a = k.vec(adv if adv is not None else env)
    if ratio is not None:
        r = k.vec(ratio)
    elif logp_new is not None and logp_old is not None:
        ln = k.vec(logp_new)
        lo = k.vec(logp_old)
        r = [math.exp(ln[i] - lo[i]) for i in range(len(ln))]
    else:
        r = k.vec(policy)
    e = float(clip_eps)
    n = len(a)
    tot = 0.0
    nclip = 0.0
    for i in range(n):
        un = r[i] * a[i]
        cl = (1.0 - e if r[i] < 1.0 - e else (1.0 + e if r[i] > 1.0 + e else r[i])) * a[i]
        if cl < un:
            tot += cl
            nclip += 1.0
        else:
            tot += un
    lclip = tot / n if n else float("nan")
    lvf = float("nan")
    if v_pred is not None and v_targ is not None:
        vp = k.vec(v_pred)
        vt = k.vec(v_targ)
        s = 0.0
        for i in range(len(vp)):
            s += (vp[i] - vt[i]) ** 2
        lvf = s / len(vp) if vp else float("nan")
    lent = k.mean(k.vec(entropy)) if entropy is not None else float("nan")
    total = lclip
    if lvf == lvf:
        total -= float(c1) * lvf
    if lent == lent:
        total += float(c2) * lent
    return RichResult(
        title="PPO clipped surrogate",
        summary_lines=[("L^CLIP", lclip)],
        payload={
            "estimate": lclip,
            "l_clip": lclip,
            "l_vf": lvf,
            "l_entropy": lent,
            "total": total,
            "frac_clipped": nclip / n if n else float("nan"),
            "n": n,
            "method": "PPO clipped surrogate objective (Schulman et al. 2017, eq. 7)",
        },
    )


def cheatsheet():
    return "ppoc: Proximal policy optimization (clipped surrogate)"
