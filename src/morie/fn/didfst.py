# morie.fn -- function file (rootcoder007/morie)
r"""Difference-in-differences with a forest: heterogeneous ATT.

The scalar difference-in-differences estimator answers one question --
how much did the treated group gain relative to the control group --
and answers it with one number. This module asks the same question
conditionally on covariates, and it does so **without** giving up the
thing that makes DiD work.

**The identification argument is unchanged; only the averaging moves.**
In the block-adoption design a panel runs :math:`t = 1, \dots, T`, all
units start untreated, and those with :math:`D_i = 1` become treated
after a shared event time :math:`H`. Writing the pre/post means

.. math:: \bar Y_i^{\text{post}} =
          \frac{1}{T-H}\sum_{t=H+1}^{T} Y_{it}, \qquad
          \bar Y_i^{\text{pre}} = \frac{1}{H}\sum_{t=1}^{H} Y_{it},

the scalar DiD estimator is the difference in group means of
:math:`\Delta_i = \bar Y_i^{\text{post}} - \bar Y_i^{\text{pre}}`,

.. math:: \hat\tau_{DID}
          = \frac{\sum_i D_i \Delta_i}{\sum_i D_i}
          - \frac{\sum_i (1-D_i) \Delta_i}{\sum_i (1-D_i)}.

That is exactly Wager's eq. (13.7): the pre-period is used to build a
baseline that is subtracted off, and the post-minus-pre differences are
then compared across adopters and non-adopters. Under non-anticipation
(Assumption 13.1) and parallel trends (Assumption 13.2) it is unbiased
for the sample ATT of eq. (13.8).

**The conditional version replaces the two group means by forest
weights.** The differenced outcome :math:`\Delta_i` is a single number
per unit, so a causal forest can be grown on :math:`(X_i, \Delta_i,
D_i)` and the same contrast taken locally,

.. math:: \hat\tau(x) =
          \frac{\sum_i \alpha_i(x) D_i \Delta_i}
               {\sum_i \alpha_i(x) D_i}
          - \frac{\sum_i \alpha_i(x) (1-D_i)\Delta_i}
                 {\sum_i \alpha_i(x)(1-D_i)},

with :math:`\alpha_i(x)` the honest-forest weights of :mod:`hntfst`
(GRF eq. 3). Setting every weight to :math:`1/n` recovers the scalar
estimator **identically**, which is the anchor's first check: the
conditional method must contain the unconditional one as a special
case, not merely approximate it.

**What parallel trends buys and what it does not.** Parallel trends is
a statement about the *never-treated* potential outcome path, so it is
untestable in the post period. What the pre-period does allow is a
placebo: split the pre-period in two and run the whole estimator inside
it, where the true effect is zero by non-anticipation. A non-zero
placebo is evidence against the assumption; a zero placebo is not
evidence for it. Both are reported and neither is dressed up as the
other.

**Staggered adoption is a different estimand, not a harder version of
the same one.** When units adopt at different times, the two-way
fixed-effects regression coefficient is a weighted average of
group-time effects whose weights can be negative, so it can have the
wrong sign even when every underlying effect shares a sign. The route
offered here is the group-time decomposition: for each adoption cohort
:math:`g` and each period :math:`t`, estimate
:math:`ATT(g,t)` against a clean comparison group -- either the
never-treated or the not-yet-treated -- and aggregate afterwards with
weights the user can see. That is Callaway and Sant'Anna's proposal;
their eq. (2.7)/(2.8) is what ``group_time_att`` computes.

References
----------
Wager, S. (2025) *Causal Inference: A Statistical Learning Approach*,
Stanford University, draft of 26 November 2025. Chapter 13
"Event-Study Designs": Definitions 13.1-13.2 (block and staggered
adoption), Assumption 13.1 (non-anticipation), eq. (13.5) (the
post-event estimand), eq. (13.7) (the DiD estimator implemented here),
eq. (13.8) (the SATT), Assumption 13.2 (parallel trends) and
Theorem 13.2 (unbiasedness of DiD for the SATT).

Callaway, B. & Sant'Anna, P. H. C. (2021) "Difference-in-Differences
with multiple time periods", *Journal of Econometrics* 225(2),
200-230, doi:10.1016/j.jeconom.2020.12.001, arXiv:1803.09015. The
group-time ATT(g,t) decomposition and the choice between never-treated
and not-yet-treated comparison groups.

Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized random
forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709. The forest weights alpha_i(x) of eq. (3) and
the honesty requirement, both supplied here by :mod:`hntfst`.

Notes
-----
The ledger recorded this module as "Athey et al (2024), DiD random
forest". No such paper was located; the citation was carried over from
the generated stub and is not the source of anything below. The
implementation follows Wager (2025) ch. 13 for the design and
Callaway-Sant'Anna (2021) for the staggered case.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest

__all__ = ["panel_differences", "did_estimate", "did_forest",
           "group_time_att", "aggregate_att", "placebo_did"]

_EPS = 1e-12
_COMPARISON = ("never-treated", "not-yet-treated")


def _panel(Y):
    """Y as an n-by-T list of lists, validated."""
    M = k.mat(Y)
    if not M:
        raise ValueError("didfst: the panel is empty")
    T = len(M[0])
    if T < 2:
        raise ValueError("didfst: need at least 2 periods, got %d" % T)
    for r, row in enumerate(M):
        if len(row) != T:
            raise ValueError("didfst: row %d has %d periods, expected "
                             "%d -- the panel must be balanced"
                             % (r, len(row), T))
    return [[float(v) for v in row] for row in M], len(M), T


def panel_differences(Y, event_time):
    r"""Post-minus-pre differences, the :math:`\Delta_i` of eq. (13.7).

    ``event_time`` is :math:`H` in Definition 13.1: periods
    :math:`1..H` are pre-treatment for every unit and :math:`H+1..T`
    post. It is given in 1-based period numbers, as the book writes it.
    """
    M, n, T = _panel(Y)
    H = int(event_time)
    if not 1 <= H < T:
        raise ValueError("didfst: event_time must satisfy 1 <= H < T = "
                         "%d, got %d" % (T, H))
    out = []
    for row in M:
        pre = sum(row[:H]) / float(H)
        post = sum(row[H:]) / float(T - H)
        out.append(post - pre)
    return out


def did_estimate(delta, D, weights=None):
    r"""The DiD contrast of eq. (13.7) under arbitrary unit weights.

    With ``weights=None`` every unit counts equally and this is the
    textbook estimator. Passing forest weights turns it into the local
    estimate at a point; passing nothing else changes.
    """
    d = [float(v) for v in k.vec(delta)]
    Dv = [float(v) for v in k.vec(D)]
    n = len(d)
    if len(Dv) != n:
        raise ValueError("didfst: %d differences but %d adoption "
                         "indicators" % (n, len(Dv)))
    for v in Dv:
        if v not in (0.0, 1.0):
            raise ValueError("didfst: D must be 0/1, got %r" % (v,))
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    if len(w) != n:
        raise ValueError("didfst: %d weights for %d units"
                         % (len(w), n))
    if any(v < 0.0 for v in w):
        raise ValueError("didfst: weights must be non-negative")
    st = sum(w[i] * Dv[i] for i in range(n))
    sc = sum(w[i] * (1.0 - Dv[i]) for i in range(n))
    if st <= _EPS or sc <= _EPS:
        raise ValueError("didfst: the comparison needs weight on both "
                         "adopters and non-adopters (treated %.3g, "
                         "control %.3g)" % (st, sc))
    mt = sum(w[i] * Dv[i] * d[i] for i in range(n)) / st
    mc = sum(w[i] * (1.0 - Dv[i]) * d[i] for i in range(n)) / sc
    return mt - mc, mt, mc, st, sc


def did_forest(Y, D, X, event_time, x_eval=None, n_trees=200,
               min_leaf=5, alpha=0.05, max_depth=12, seed=0,
               kind="double-sample", clusters=None):
    r"""Heterogeneous ATT: DiD taken locally under forest weights.

    Parameters
    ----------
    Y : array-like
        The balanced panel, n rows by T periods.
    D : array-like
        Adoption indicator, 1 for units treated after ``event_time``.
    X : array-like
        Covariates, n rows.
    event_time : int
        :math:`H`, in 1-based period numbers.
    x_eval : array-like, optional
        Points at which to report :math:`\hat\tau(x)`. Defaults to the
        training rows, which gives the fitted CATT for every unit.

    Returns
    -------
    RichResult
        ``estimate`` is the average of the local estimates,
        ``tau`` the per-point values, and ``att_uniform`` the scalar
        eq. (13.7) estimator for comparison.
    """
    delta = panel_differences(Y, event_time)
    Xm = k.mat(X)
    n = len(delta)
    if len(Xm) != n:
        raise ValueError("didfst: %d covariate rows for %d panel units"
                         % (len(Xm), n))
    Dv = [float(v) for v in k.vec(D)]
    flat, _, _, _, _ = did_estimate(delta, Dv)
    trees, bags, s = grow_forest(Xm, delta, W=Dv, kind=kind,
                                 n_trees=n_trees, min_leaf=min_leaf,
                                 alpha=alpha, max_depth=max_depth,
                                 seed=seed, clusters=clusters)
    pts = Xm if x_eval is None else k.mat(x_eval)
    taus, wt_t, wt_c = [], [], []
    for x in pts:
        w = forest_weights(trees, Xm, x)
        t, _, _, st, sc = did_estimate(delta, Dv, weights=w)
        taus.append(t)
        wt_t.append(st)
        wt_c.append(sc)
    return RichResult(payload={
        "estimate": sum(taus) / len(taus),
        "tau": taus, "delta": delta,
        "att_uniform": flat,
        "treated_weight": wt_t, "control_weight": wt_c,
        "n": n, "n_trees": int(n_trees), "event_time": int(event_time),
        "design": "block-adoption",
        "method": "difference-in-differences under honest forest "
                  "weights; Wager (2025) eq. (13.7) localised by "
                  "Athey-Tibshirani-Wager (2019) eq. (3)",
    })


def placebo_did(Y, D, event_time, split=None):
    r"""Run the estimator entirely inside the pre-period.

    Under non-anticipation (Assumption 13.1) the true effect here is
    zero, so a non-zero estimate is evidence against parallel trends. A
    zero estimate is **not** evidence for it -- parallel trends
    constrains the post-period path, which no pre-period test can see.
    """
    M, n, T = _panel(Y)
    H = int(event_time)
    if H < 2:
        raise ValueError("didfst: a pre-period placebo needs at least "
                         "2 pre-periods, event_time is %d" % H)
    cut = H // 2 if split is None else int(split)
    if not 1 <= cut < H:
        raise ValueError("didfst: the placebo split must satisfy "
                         "1 <= split < %d, got %d" % (H, cut))
    pre = [row[:H] for row in M]
    d = panel_differences(pre, cut)
    est, mt, mc, _, _ = did_estimate(d, D)
    return RichResult(payload={
        "estimate": est, "treated_change": mt, "control_change": mc,
        "split": cut, "n_pre": H,
        "interpretation": "zero is consistent with parallel trends but "
                          "does not establish it",
        "method": "pre-period placebo DiD; Wager (2025) Assumption 13.1",
    })


def group_time_att(Y, first_treated, comparison="not-yet-treated"):
    r"""Callaway-Sant'Anna :math:`ATT(g,t)` under staggered adoption.

    ``first_treated[i]`` is the 1-based period at which unit *i* first
    becomes treated, or ``None``/``inf`` for never-treated. For a
    cohort :math:`g` and a period :math:`t \ge g`, the estimate
    compares the change from :math:`g-1` to :math:`t` between cohort
    :math:`g` and the comparison group,

    .. math:: \widehat{ATT}(g,t) =
              \big(\bar Y_t^{g} - \bar Y_{g-1}^{g}\big)
              - \big(\bar Y_t^{C} - \bar Y_{g-1}^{C}\big).

    ``comparison="never-treated"`` uses only units that never adopt --
    cleaner, but unusable when everyone eventually adopts.
    ``comparison="not-yet-treated"`` additionally uses units whose own
    adoption is still in the future at :math:`t`, which is the larger
    and usually more precise comparison group. Both are offered because
    the paper offers both and they are not interchangeable.
    """
    M, n, T = _panel(Y)
    if comparison not in _COMPARISON:
        raise ValueError("didfst: comparison must be one of %s, got %r"
                         % (", ".join(_COMPARISON), comparison))
    if len(first_treated) != n:
        raise ValueError("didfst: %d adoption times for %d units"
                         % (len(first_treated), n))
    G = []
    for v in first_treated:
        if v is None:
            G.append(None)
            continue
        f = float(v)
        if f != f or f == float("inf"):
            G.append(None)
            continue
        g = int(f)
        if not 2 <= g <= T:
            raise ValueError("didfst: adoption time %d is outside "
                             "2..T = %d (a unit treated in period 1 "
                             "has no pre-period)" % (g, T))
        G.append(g)
    cohorts = sorted({g for g in G if g is not None})
    if not cohorts:
        raise ValueError("didfst: no unit is ever treated")
    out = {}
    for g in cohorts:
        idx_g = [i for i in range(n) if G[i] == g]
        for t in range(g, T + 1):
            if comparison == "never-treated":
                idx_c = [i for i in range(n) if G[i] is None]
            else:
                idx_c = [i for i in range(n)
                         if G[i] is None or G[i] > t]
            if not idx_c:
                continue
            a, b = t - 1, g - 2          # 0-based period indices
            dg = (sum(M[i][a] for i in idx_g) / len(idx_g)
                  - sum(M[i][b] for i in idx_g) / len(idx_g))
            dc = (sum(M[i][a] for i in idx_c) / len(idx_c)
                  - sum(M[i][b] for i in idx_c) / len(idx_c))
            out[(g, t)] = {"att": dg - dc, "n_treated": len(idx_g),
                           "n_control": len(idx_c)}
    if not out:
        raise ValueError("didfst: no (g, t) cell had a usable "
                         "comparison group")
    return RichResult(payload={
        "att": out, "cohorts": cohorts, "T": T, "n": n,
        "comparison": comparison,
        "estimate": sum(v["att"] for v in out.values()) / len(out),
        "method": "group-time ATT(g,t), Callaway & Sant'Anna (2021)",
    })


def aggregate_att(gt, scheme="simple", horizon=None):
    r"""Aggregate :math:`ATT(g,t)` with weights the caller can see.

    ``"simple"``
        cohort-size weighted average over all treated cells.
    ``"event"``
        by event time :math:`e = t - g`, which is the dynamic profile;
        ``horizon`` restricts to :math:`e \le` horizon.
    ``"cohort"``
        one number per adoption cohort, averaged over its post periods.
    """
    if scheme not in ("simple", "event", "cohort"):
        raise ValueError("didfst: scheme must be simple, event or "
                         "cohort, got %r" % (scheme,))
    cells = gt["att"] if isinstance(gt, (dict, RichResult)) else gt
    if not cells:
        raise ValueError("didfst: nothing to aggregate")
    if scheme == "simple":
        num = sum(v["att"] * v["n_treated"] for v in cells.values())
        den = sum(v["n_treated"] for v in cells.values())
        return {"estimate": num / den, "scheme": "simple", "n_cells":
                len(cells)}
    keyed = {}
    for (g, t), v in cells.items():
        key = (t - g) if scheme == "event" else g
        if scheme == "event" and horizon is not None and key > horizon:
            continue
        keyed.setdefault(key, []).append(v)
    if not keyed:
        raise ValueError("didfst: the horizon excluded every cell")
    prof = {}
    for key, vs in keyed.items():
        num = sum(v["att"] * v["n_treated"] for v in vs)
        den = sum(v["n_treated"] for v in vs)
        prof[key] = num / den
    return {"profile": prof, "scheme": scheme,
            "estimate": sum(prof.values()) / len(prof)}


def cheatsheet():
    return ("didfst: DiD forest. Delta_i = post-mean - pre-mean; the "
            "scalar estimator (Wager 2025 eq. 13.7) is the difference "
            "of group means of Delta, and the forest version is the "
            "SAME contrast under alpha_i(x) weights -- uniform weights "
            "reproduce it exactly. Parallel trends is untestable in "
            "the post period; placebo_did only checks the pre-period. "
            "Staggered adoption is NOT the same estimand: use "
            "group_time_att (Callaway-Sant'Anna 2021), because TWFE "
            "weights can go negative.")


# compact alias per ledger/NAMING.md
didforest = did_forest
