# morie.fn -- function file (rootcoder007/morie)
r"""Honest DiD: what survives if parallel trends is only approximate.

A pre-trends test is not a licence. It has low power against exactly
the violations that matter, and conditioning the analysis on having
passed it distorts the inference that follows. The alternative is to
stop asserting parallel trends and instead ask: *how large a violation
would it take to overturn the conclusion?*

**The decomposition.** An event study delivers coefficients
:math:`\hat\beta \in \mathbb R^{\underline T + \bar T}` split into pre-
and post-treatment blocks, with

.. math:: \beta = \tau + \delta, \qquad \tau_{\text{pre}} = 0,

so the pre-period coefficients estimate the *violation* :math:`\delta`
and nothing else, while the post-period coefficients are contaminated
by it (Assumption 1). Parallel trends is the special case
:math:`\delta = 0`. Rather than assume that, restrict :math:`\delta`
to a set :math:`\Delta` and report the whole identified set for the
target parameter :math:`\theta = l' \tau_{\text{post}}`.

**Two families of restriction, and they say different things.**

``"SD"`` -- :math:`\Delta^{SD}(M)` bounds the *second* differences of
:math:`\delta` by :math:`M`:

.. math:: \left|(\delta_{t+1}-\delta_t)-(\delta_t-\delta_{t-1})\right|
          \le M \quad \text{for all } t.

:math:`M = 0` is exactly "the differential trend is linear", the
common practice of extrapolating a straight line from the pre-period.
:math:`M > 0` allows the trend to bend, by at most :math:`M` per
period. This is a *smoothness* statement.

``"RM"`` -- :math:`\Delta^{RM}(\bar M)` bounds the post-treatment
violation relative to what was actually observed before:

.. math:: |\delta_t| \le \bar M \cdot
          \max_{s<0}|\delta_{s+1}-\delta_s|,

so :math:`\bar M = 1` says "the post-period violation is no larger
than the largest pre-period one". This is a *relative magnitude*
statement, and it is the one that makes the pre-period informative
without pretending a passed test proves anything.

**The breakdown value is the honest summary.** Rather than a single
interval, report the largest :math:`M` (or :math:`\bar M`) at which
the conclusion -- typically :math:`\theta > 0` -- still holds. That
number is what a reader can argue with: it converts "parallel trends
holds" into "the violation would have to exceed this to change the
answer".

**Why the identified set is an interval and how it is computed.** For
these polyhedral :math:`\Delta`, the set of :math:`\theta` consistent
with :math:`\beta` is
:math:`\{l'(\beta_{\text{post}} - \delta_{\text{post}}) : \delta \in
\Delta,\ \delta_{\text{pre}} = \beta_{\text{pre}}\}`, whose endpoints
are the min and max of a linear objective over a polyhedron. Under
:math:`\Delta^{SD}` the constraint set is a bounded polytope in the
free post-period coordinates, and the extreme points are attained by
pushing the second differences to :math:`\pm M`, so the endpoints have
a closed form the implementation uses directly and the anchor checks
against a brute-force grid search.

References
----------
Rambachan, A. & Roth, J. (2023) "A More Credible Approach to Parallel
Trends", *The Review of Economic Studies* 90(5), 2555-2591,
doi:10.1093/restud/rdad018. Sec. 2.1 (event-study coefficients, incl.
Example 2 for staggered designs), Assumption 1 (the tau/delta
decomposition), Sec. 2.3 (Delta^SD and Delta^RM), Sec. 3 (the
identified set and fixed-length confidence sets), and the breakdown
value.

Callaway, B. & Sant'Anna, P. H. C. (2021) "Difference-in-Differences
with multiple time periods", *Journal of Econometrics* 225(2),
200-230, doi:10.1016/j.jeconom.2020.12.001. The staggered-adoption
event-study coefficients this sensitivity analysis is applied to in
Example 2.

Wager, S. (2025) *Causal Inference: A Statistical Learning Approach*,
Stanford University, draft of 26 November 2025. Chapter 13, Assumption
13.2, for the parallel-trends assumption being relaxed here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["identified_set", "sensitivity_curve", "breakdown_value",
           "fixed_length_ci"]

_EPS = 1e-12
_FAMILIES = ("SD", "RM")


def _split(beta, n_pre, n_post):
    b = [float(v) for v in k.vec(beta)]
    if len(b) != int(n_pre) + int(n_post):
        raise ValueError("snmtst: %d coefficients but n_pre + n_post "
                         "= %d" % (len(b), int(n_pre) + int(n_post)))
    if int(n_pre) < 1 or int(n_post) < 1:
        raise ValueError("snmtst: need at least one pre and one post "
                         "period, got %d and %d" % (n_pre, n_post))
    return b[:int(n_pre)], b[int(n_pre):]


def _target(post, l_vec):
    if l_vec is None:
        return [1.0] + [0.0] * (len(post) - 1)
    lv = [float(v) for v in k.vec(l_vec)]
    if len(lv) != len(post):
        raise ValueError("snmtst: the target vector has %d entries "
                         "for %d post periods" % (len(lv), len(post)))
    return lv


def identified_set(beta, n_pre, n_post, M=0.0, family="SD",
                   l_vec=None, grid=None):
    r"""The set of :math:`\theta = l'\tau_{\text{post}}` consistent
    with :math:`\delta \in \Delta`.

    ``family="SD"`` uses :math:`\Delta^{SD}(M)`, ``"RM"`` uses
    :math:`\Delta^{RM}(M)`. ``M = 0`` under ``"SD"`` is the linear
    extrapolation practitioners already do, so the returned interval
    collapses to a point there -- which is the check that the linear
    special case has not been lost.

    The endpoints are obtained by maximising and minimising the linear
    objective over the polyhedron. Set ``grid`` to an integer to solve
    the same problem by brute force over a lattice instead; the two
    agree, and the option exists so the closed form can be falsified.
    """
    if family not in _FAMILIES:
        raise ValueError("snmtst: family must be SD or RM, got %r"
                         % (family,))
    Mv = float(M)
    if Mv < 0.0:
        raise ValueError("snmtst: M must be non-negative, got %r"
                         % (M,))
    pre, post = _split(beta, n_pre, n_post)
    lv = _target(post, l_vec)
    Tp = len(post)

    if family == "SD":
        # delta_pre is pinned at beta_pre. Extrapolate linearly from
        # the last two pre-period values and let the second difference
        # wander within +-M each step.
        if len(pre) < 2:
            raise ValueError("snmtst: Delta^SD needs at least 2 "
                             "pre-periods to define a slope, got %d"
                             % len(pre))
        base, slope = pre[-1], pre[-1] - pre[-2]
        # the linear path: delta_t = base + slope * t
        lin = [base + slope * (t + 1) for t in range(Tp)]
        # Write e_t = delta_t - lin_t. Since lin has zero second
        # difference, the constraint is on e directly:
        #     e_{t+1} - 2 e_t + e_{t-1} = r_t,   |r_t| <= M,
        # with e pinned at 0 for the last two pre-periods. Solving the
        # recursion forwards from e_{-1} = e_0 = 0 gives
        #     e_t = sum_{j=1}^{t} (t - j + 1) r_j,
        # where t counts post-periods from 1. Indexing post-periods
        # from 0 as the code does, the step number is t+1, so the
        # weight on r_j is (t + 1) - j + 1 = t - j + 2.
        # so e is LINEAR in the free second differences r. The
        # objective l'(beta_post - delta_post) is therefore linear in r
        # over the box |r_j| <= M, and its extremes are
        #     point -+ M * sum_j |c_j|,   c_j = sum_{t>=j} l_t (t-j+1).
        # Note this is NOT sum_t |l_t| * max|e_t|: the e_t share the
        # same r and cannot be pushed to their individual extremes at
        # once unless l has a single non-zero entry.
        dev = [Mv * (t + 1) * (t + 2) / 2.0 for t in range(Tp)]
        c = [sum(lv[t] * (t - j + 2) for t in range(j - 1, Tp))
             for j in range(1, Tp + 1)]
        point = sum(lv[t] * (post[t] - lin[t]) for t in range(Tp))
        if grid is None:
            half = Mv * sum(abs(v) for v in c)
            lo, hi = point - half, point + half
        else:
            lo, hi = _brute(point, c, Mv, int(grid), post=post,
                            lin=lin, lv=lv)
        return {"lower": lo, "upper": hi, "estimate": point,
                "linear_path": lin, "max_deviation": dev,
                "coefficients": c,
                "M": Mv, "family": "SD",
                "width": hi - lo,
                "note": "M = 0 is exactly linear extrapolation from "
                        "the pre-period"}

    # RM: |delta_t| <= M * max pre-period first difference
    if len(pre) < 2:
        raise ValueError("snmtst: Delta^RM needs at least 2 "
                         "pre-periods to form a first difference, got "
                         "%d" % len(pre))
    scale = max(abs(pre[i + 1] - pre[i]) for i in range(len(pre) - 1))
    bound = Mv * scale
    hi = sum(lv[t] * post[t] for t in range(Tp)) \
        + bound * sum(abs(v) for v in lv)
    lo = sum(lv[t] * post[t] for t in range(Tp)) \
        - bound * sum(abs(v) for v in lv)
    return {"lower": lo, "upper": hi,
            "estimate": sum(lv[t] * post[t] for t in range(Tp)),
            "pre_max_change": scale, "bound": bound,
            "M": Mv, "family": "RM", "width": hi - lo,
            "note": "M = 1 says the post violation is no larger than "
                    "the largest observed pre-period change"}


def _brute(point, c, M, grid, post=None, lin=None, lv=None):
    """Min/max over a lattice of the free second differences r.

    An INDEPENDENT route: rather than reusing the coefficients ``c``,
    it runs the second-difference recursion
    :math:`e_{t+1} = 2e_t - e_{t-1} + r_t` forward from
    :math:`e_{-1} = e_0 = 0` and evaluates the objective directly. If
    the algebra behind ``c`` were wrong the two would disagree, which
    is the point of having it.
    """
    if grid < 2:
        raise ValueError("snmtst: the brute-force grid needs at least "
                         "2 points per coordinate, got %d" % grid)
    p = len(c)
    if grid ** p > 2000000:
        raise ValueError("snmtst: a %d-point grid over %d coordinates "
                         "is %d evaluations -- refuse rather than hang"
                         % (grid, p, grid ** p))
    steps = [-M + 2.0 * M * j / (grid - 1) for j in range(grid)]
    idx = [0] * p
    lo = hi = None
    direct = post is not None and lin is not None and lv is not None
    while True:
        if direct:
            r = [steps[idx[j]] for j in range(p)]
            e_prev2, e_prev = 0.0, 0.0
            e = []
            for t in range(p):
                cur = 2.0 * e_prev - e_prev2 + r[t]
                e.append(cur)
                e_prev2, e_prev = e_prev, cur
            val = sum(lv[t] * (post[t] - (lin[t] + e[t]))
                      for t in range(p))
        else:
            val = point - sum(c[j] * steps[idx[j]] for j in range(p))
        lo = val if lo is None else min(lo, val)
        hi = val if hi is None else max(hi, val)
        q = p - 1
        while q >= 0:
            idx[q] += 1
            if idx[q] < grid:
                break
            idx[q] = 0
            q -= 1
        if q < 0:
            break
    return lo, hi


def sensitivity_curve(beta, n_pre, n_post, Ms, family="SD",
                      l_vec=None):
    """The identified set as a function of the relaxation parameter.

    The width is non-decreasing in ``M`` by construction -- a larger
    :math:`\\Delta` cannot shrink the set -- which is one of the
    properties the anchor checks.
    """
    out = []
    for M in Ms:
        s = identified_set(beta, n_pre, n_post, M=M, family=family,
                           l_vec=l_vec)
        out.append({"M": s["M"], "lower": s["lower"],
                    "upper": s["upper"], "width": s["width"]})
    return {"curve": out, "family": family,
            "M": [o["M"] for o in out],
            "width": [o["width"] for o in out]}


def breakdown_value(beta, n_pre, n_post, family="SD", l_vec=None,
                    sign="positive", M_max=10.0, tol=1e-9):
    r"""The largest relaxation at which the sign conclusion survives.

    With ``sign="positive"`` this is
    :math:`\sup\{M : \text{lower bound of the identified set} > 0\}`.
    Found by bisection, which is valid because the bound is monotone
    in :math:`M`. Returns 0 if the conclusion already fails under
    parallel trends itself, and ``M_max`` if it survives everywhere on
    the search range -- both reported explicitly rather than as a
    silent endpoint.
    """
    if sign not in ("positive", "negative"):
        raise ValueError("snmtst: sign must be positive or negative, "
                         "got %r" % (sign,))

    def holds(M):
        s = identified_set(beta, n_pre, n_post, M=M, family=family,
                           l_vec=l_vec)
        return s["lower"] > 0.0 if sign == "positive" \
            else s["upper"] < 0.0

    if not holds(0.0):
        return {"breakdown": 0.0, "family": family, "sign": sign,
                "status": "the conclusion fails even at M = 0"}
    if holds(float(M_max)):
        return {"breakdown": float(M_max), "family": family,
                "sign": sign,
                "status": "survives the whole search range; increase "
                          "M_max to find the true breakdown"}
    lo, hi = 0.0, float(M_max)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if holds(mid):
            lo = mid
        else:
            hi = mid
    return {"breakdown": lo, "family": family, "sign": sign,
            "status": "interior"}


def fixed_length_ci(beta, sigma, n_pre, n_post, M=0.0, family="SD",
                    l_vec=None, level=0.95):
    r"""A confidence set for :math:`\theta` that covers the identified
    set, adding sampling uncertainty to the partial-identification
    width.

    ``sigma`` is the standard error of :math:`l'\hat\beta_{\text{post}}`
    -- the sampling noise in the estimate, separate from the
    identification width. The interval returned is the identified set
    widened by the normal critical value, which is conservative and is
    labelled as such: the exact fixed-length construction of Sec. 3
    solves a further optimisation that this does not attempt.
    """
    s = identified_set(beta, n_pre, n_post, M=M, family=family,
                       l_vec=l_vec)
    if float(sigma) < 0.0:
        raise ValueError("snmtst: sigma must be non-negative, got %r"
                         % (sigma,))
    if not 0.0 < float(level) < 1.0:
        raise ValueError("snmtst: level must be in (0, 1), got %r"
                         % (level,))
    z = k.qnorm(0.5 + float(level) / 2.0)
    return RichResult(payload={
        "estimate": s["estimate"],
        "lower": s["lower"] - z * float(sigma),
        "upper": s["upper"] + z * float(sigma),
        "identified_lower": s["lower"], "identified_upper": s["upper"],
        "M": s["M"], "family": family, "level": float(level),
        "conservative": True,
        "method": "identified set (Rambachan & Roth 2023 Sec. 2.3) "
                  "widened by the normal critical value; the exact "
                  "fixed-length construction of their Sec. 3 is "
                  "tighter",
    })


def cheatsheet():
    return ("snmtst: honest DiD. beta = tau + delta with tau_pre = 0, "
            "so the PRE coefficients estimate the violation. Instead "
            "of assuming delta = 0, bound it: Delta^SD(M) caps the "
            "SECOND differences (M=0 IS linear extrapolation), "
            "Delta^RM(Mbar) caps the post violation at Mbar times the "
            "largest pre-period change. Report the BREAKDOWN value -- "
            "the M at which the sign flips -- not a pre-trends "
            "p-value, which has low power exactly where it matters.")


# compact alias per ledger/NAMING.md
sensitivitytest = breakdown_value

# public names resolved by fn/_lazy_map.json
sensitivity_did = breakdown_value
sensitivitydid = breakdown_value
