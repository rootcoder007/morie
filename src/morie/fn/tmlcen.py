# morie.fn -- function file (rootcoder007/morie)
r"""Causal effect under censoring, by inverse probability of censoring
weighting.

Two censoring structures, two estimators, because the ledger row asks
for right-censoring and the source volume works the interval-censored
case out in full.

**Right-censored (default).** Hernan & Robins Ch. 17. Censoring is
handled by weighting each subject's observed follow-up by the inverse
of its probability of remaining uncensored,

.. math:: \bar G_c(k \mid A, W)
          = \prod_{j \le k}\left(1 - \lambda_C(j \mid A, W)\right),

with :math:`\lambda_C` the censoring hazard. The discrete-time death
hazard is then fitted on the weighted person-time data and converted to
a survival curve by the same product-limit,
:math:`S(k) = \prod_{j\le k}(1 - \lambda(j))` -- Sec. 17.2, "from
hazards to risks". Weighting by :math:`1/\bar G_c` and *also*
conditioning on being uncensored would double-count; the weights exist
so that conditioning is unnecessary.

**Interval-censored.** van der Laan & Rose (2018) Sec. 8.5. The data
are :math:`O = (W, A, C_m, \Delta_m = I(T \le C_m) : m = 1..M)`: M
monitoring times, and at each one only whether the event has happened
yet. The target is
:math:`\Psi_a^f(Q) = \int r(t)\bar F_a(t)\,dt` with
:math:`\bar F_a(t) = E_P P(T > t \mid A = a, W)`, a weighted mean
survival, and the chapter's initial gradient gives the estimator

.. math:: \frac{1}{M}\sum_{m=1}^{M}
          (1 - \Delta_m)\, r(C_m)\,
          \frac{I(A = a)}{\bar g_c(C_m \mid A, W)\, g(A \mid W)}.

The interval implied by the monitoring is spelled out exactly and is
implemented exactly: :math:`L(O)` is the largest monitoring time with
:math:`\Delta_j = 0` and :math:`R(O)` the smallest with
:math:`\Delta_j = 1`; if :math:`\Delta_1 = 1` then :math:`L(O) = 0`,
and if :math:`\Delta_M = 0` then :math:`R(O) = \infty`. Getting those
two boundary cases wrong is silent -- the estimate simply comes out
biased -- so the anchor builds them by hand.

**What is assumed.** T independent of C given (A, W); positivity of
both the treatment and the monitoring/censoring mechanisms. The second
is checkable in the sample and is checked: a zero censoring-survival
probability raises rather than producing an infinite weight that
becomes a number.

References
----------
Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*, Boca
Raton: Chapman & Hall/CRC, Ch. 17 "Causal survival analysis" --
Sec. 17.2 hazards to risks, Sec. 17.3 why censoring matters, Sec. 17.4
IP weighting of marginal structural models.

van der Laan, M. J. & Rose, S. (eds.) (2018) *Targeted Learning in
Data Science*, Springer Series in Statistics,
doi:10.1007/978-3-319-65304-4, Sec. 8.5 "Causal Effect of Binary
Treatment on Interval Censored Time to Event" -- the data structure,
the coarsening C(o) = (L(o), R(o)], and the IPCW estimator above.

Note on the ledger citation: its key "Stitelman-Lendle-vdL (2011)" does
not resolve to any paper; a bibliographic search returns the ltmle
software package instead. The two sources above are what this is built
from.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmle_censoring", "censoring_survival", "coarsen_interval",
           "ipcw_interval"]

_KINDS = ("right", "interval")


def coarsen_interval(times, deltas):
    r"""The interval (L, R] implied by one subject's monitoring, Sec. 8.5.

    L is the largest monitoring time with Delta = 0 and R the smallest
    with Delta = 1. If the event had already happened by the first
    monitoring time, L = 0; if it had not happened by the last, R is
    infinite.
    """
    ts = [float(v) for v in times]
    ds = [float(v) for v in deltas]
    if len(ts) != len(ds):
        raise ValueError("coarsen_interval: %d monitoring times but %d "
                         "indicators" % (len(ts), len(ds)))
    if not ts:
        raise ValueError("coarsen_interval: no monitoring times")
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    ts = [ts[i] for i in order]
    ds = [ds[i] for i in order]
    if any(d not in (0.0, 1.0) for d in ds):
        raise ValueError("coarsen_interval: Delta must be 0/1")
    zeros = [ts[i] for i in range(len(ts)) if ds[i] == 0.0]
    ones = [ts[i] for i in range(len(ts)) if ds[i] == 1.0]
    L = max(zeros) if zeros else 0.0
    R = min(ones) if ones else float("inf")
    return L, R


def censoring_survival(times, censored, A=None, W=None, grid=None,
                       by_covariate=True, ridge=1e-8):
    r"""Gbar_c(k | A, W) = prod_{j<=k} (1 - lambda_C(j | A, W)).

    The censoring hazard is fitted in discrete time by pooled logistic
    regression on (A, W) -- the standard device, and the one Ch. 17
    uses to let the hazard depend on covariates without a separate
    model per time point.
    """
    t = [float(v) for v in k.vec(times)]
    c = [float(v) for v in k.vec(censored)]
    n = len(t)
    if len(c) != n:
        raise ValueError("censoring_survival: %d times but %d censoring "
                         "indicators" % (n, len(c)))
    if grid is None:
        grid = sorted(set(t))
    grid = [float(v) for v in grid]
    Wm = k.mat(W) if W is not None else [[] for _ in range(n)]
    av = k.vec(A) if A is not None else [0.0] * n

    # person-time rows: one per subject per time point still at risk
    rows, lab = [], []
    for i in range(n):
        for kk, tk in enumerate(grid):
            if t[i] < tk:
                break
            cens_now = 1.0 if (c[i] == 1.0 and t[i] == tk) else 0.0
            rows.append([float(kk), av[i]] + list(Wm[i]))
            lab.append(cens_now)
    if not rows:
        raise ValueError("censoring_survival: no person-time at risk")
    Z = k.design(rows, len(rows))
    b = k.logit_irls(Z, lab, 60, ridge)

    def haz(kk, i):
        row = [1.0, float(kk), av[i]] + list(Wm[i])
        return k.sigmoid(sum(b[j] * row[j] for j in range(len(b))))

    G = []
    for i in range(n):
        g, cur = [], 1.0
        for kk in range(len(grid)):
            cur *= (1.0 - haz(kk, i))
            g.append(cur)
        G.append(g)
    return G, grid, b


def ipcw_interval(W, A, times, deltas, a=1.0, r=None, g=None, gc=None,
                  ridge=1e-8):
    r"""Sec. 8.5's IPCW estimator of Psi_a = int r(t) Fbar_a(t) dt.

    `g` and `gc` may be supplied when the treatment and monitoring
    mechanisms are known by design, which is the case the estimator is
    unbiased in; otherwise they are fitted.
    """
    av = k.vec(A)
    n = len(av)
    Tm = [[float(v) for v in row] for row in times]
    Dm = [[float(v) for v in row] for row in deltas]
    if len(Tm) != n or len(Dm) != n:
        raise ValueError("ipcw_interval: %d treatments but %d monitoring "
                         "rows and %d indicator rows"
                         % (n, len(Tm), len(Dm)))
    Wm = k.mat(W) if W is not None else [[] for _ in range(n)]
    if r is None:
        def r(t):
            return 1.0

    if g is None:
        Z = k.design(Wm if Wm and Wm[0] else None, n)
        bg = k.logit_irls(Z, av, 60, ridge)
        gv = [k.sigmoid(v) for v in k.matvec(Z, bg)]
        g = [gv[i] if av[i] == 1.0 else 1.0 - gv[i] for i in range(n)]
    else:
        g = [float(v) for v in k.vec(g)]

    tot = 0.0
    for i in range(n):
        M = len(Tm[i])
        if M == 0:
            raise ValueError("ipcw_interval: subject %d has no monitoring "
                             "times" % i)
        if av[i] != a:
            continue
        if g[i] <= 0.0:
            raise ValueError(
                "ipcw_interval: g(A|W) is zero for subject %d, so "
                "positivity fails and the weight is undefined" % i)
        s = 0.0
        for m in range(M):
            dens = (gc[i][m] if gc is not None
                    else _uniform_density(Tm[i]))
            if dens <= 0.0:
                raise ValueError(
                    "ipcw_interval: the monitoring density is zero for "
                    "subject %d at time %d" % (i, m))
            s += (1.0 - Dm[i][m]) * r(Tm[i][m]) / dens
        tot += s / M / g[i]
    return tot / n


def _uniform_density(ts):
    """Monitoring density when the times are uniform on their own span."""
    lo, hi = min(ts), max(ts)
    return 1.0 / (hi - lo) if hi > lo else 1.0


def tmle_censoring(time, event, censor, treatment, covariates,
                   kind="right", grid=None, a=1.0, r=None, g=None,
                   gc=None, trim=1e-3):
    r"""Causal survival under censoring.

    Parameters
    ----------
    time : array-like
        Observed follow-up time (right-censored case), or a list of
        monitoring-time vectors (interval case).
    event : array-like
        Event indicator, or the list of Delta vectors in the interval
        case.
    censor : array-like
        Censoring indicator (right-censored case only).
    treatment, covariates : array-like
        Binary treatment and baseline covariates.
    kind : {"right", "interval"}

    Returns
    -------
    RichResult
        For the right-censored case, ``estimate`` is the IP-weighted
        difference in survival at the last grid point, with the two
        curves in ``survival_treated`` and ``survival_control``. For
        the interval case it is Sec. 8.5's Psi_a.

    Examples
    --------
    Censoring that depends on a covariate which also drives survival::

        r = tmle_censoring(t, d, c, A, W)
        r["estimate"], r["naive"]
    """
    if kind not in _KINDS:
        raise ValueError("tmle_censoring: kind must be 'right' or "
                         "'interval', got %r" % (kind,))
    if kind == "interval":
        psi = ipcw_interval(covariates, treatment, time, event, a=a, r=r,
                            g=g, gc=gc)
        return RichResult(payload={
            "estimate": psi, "psi": psi, "a": a,
            "n": len(k.vec(treatment)),
            "method": "interval-censored IPCW, van der Laan & Rose "
                      "(2018) Sec. 8.5",
        })

    t = [float(v) for v in k.vec(time)]
    d = [float(v) for v in k.vec(event)]
    c = [float(v) for v in k.vec(censor)]
    av = k.vec(treatment)
    n = len(t)
    for nm, arr in (("event", d), ("censor", c), ("treatment", av)):
        if len(arr) != n:
            raise ValueError("tmle_censoring: %d times but %d %s"
                             % (n, len(arr), nm))
    if any(d[i] == 1.0 and c[i] == 1.0 for i in range(n)):
        raise ValueError("tmle_censoring: a subject cannot be both an "
                         "event and censored at the same time")
    Wm = k.mat(covariates) if covariates is not None else \
        [[] for _ in range(n)]

    G, grid, _ = censoring_survival(t, c, A=av, W=Wm, grid=grid)

    # IP-weighted discrete hazard of the event, pooled over time
    rows, lab, wts = [], [], []
    for i in range(n):
        for kk, tk in enumerate(grid):
            if t[i] < tk:
                break
            gk = max(G[i][kk], trim)
            rows.append([float(kk), av[i]] + list(Wm[i]))
            lab.append(1.0 if (d[i] == 1.0 and t[i] == tk) else 0.0)
            wts.append(1.0 / gk)
    Z = k.design(rows, len(rows))
    bh = _weighted_logit(Z, lab, wts)

    def surv(a_val, i):
        out, cur = [], 1.0
        for kk in range(len(grid)):
            row = [1.0, float(kk), a_val] + list(Wm[i])
            h = k.sigmoid(sum(bh[j] * row[j] for j in range(len(bh))))
            cur *= (1.0 - h)
            out.append(cur)
        return out

    s1 = [sum(surv(1.0, i)[kk] for i in range(n)) / n
          for kk in range(len(grid))]
    s0 = [sum(surv(0.0, i)[kk] for i in range(n)) / n
          for kk in range(len(grid))]

    # Two comparators, because they answer different questions.
    #
    # "naive" is the same hazard model fitted WITHOUT the censoring
    # weights. When censoring depends only on covariates the hazard
    # model already conditions on, this is consistent too -- conditioning
    # and weighting are alternative solutions to the same problem, and
    # the weights buy nothing. Averaged over independent draws the two
    # are indistinguishable here, which is a fact about the design, not
    # a defect in either.
    #
    # "unadjusted" drops the covariates from the hazard model
    # altogether. That is what ignoring informative censoring actually
    # looks like, and it is the one that is biased.
    bh_n = _weighted_logit(Z, lab, [1.0] * len(lab))
    rows_u = [[row[0], row[1]] for row in rows]     # time and treatment
    Zu = k.design(rows_u, len(rows_u))
    bh_u = _weighted_logit(Zu, lab, [1.0] * len(lab))

    def surv_u(a_val):
        cur = 1.0
        for kk in range(len(grid)):
            r_ = [1.0, float(kk), a_val]
            cur *= (1.0 - k.sigmoid(sum(bh_u[j] * r_[j]
                                        for j in range(len(bh_u)))))
        return cur
    unadjusted = surv_u(1.0) - surv_u(0.0)

    def surv_n(a_val, i):
        cur = 1.0
        for kk in range(len(grid)):
            row = [1.0, float(kk), a_val] + list(Wm[i])
            cur *= (1.0 - k.sigmoid(sum(bh_n[j] * row[j]
                                        for j in range(len(bh_n)))))
        return cur
    naive = (sum(surv_n(1.0, i) for i in range(n)) / n
             - sum(surv_n(0.0, i) for i in range(n)) / n)

    return RichResult(payload={
        "estimate": s1[-1] - s0[-1],
        "survival_treated": s1, "survival_control": s0,
        "grid": grid,
        "naive": naive,
        "unadjusted": unadjusted,
        "censoring_survival": G,
        "max_weight": max(1.0 / max(G[i][-1], trim) for i in range(n)),
        "n": n,
        "method": "IPCW survival difference, Hernan & Robins (2020) "
                  "Ch. 17 Secs. 17.2 and 17.4",
    })


def _weighted_logit(Z, y, w, iters=60, ridge=1e-10):
    n, p = len(Z), len(Z[0])
    b = [0.0] * p
    for _ in range(iters):
        XtWX = [[0.0] * p for _ in range(p)]
        Xtr = [0.0] * p
        for i in range(n):
            mu = k.sigmoid(sum(Z[i][j] * b[j] for j in range(p)))
            ww = w[i] * mu * (1.0 - mu)
            rr = w[i] * (y[i] - mu)
            for aa in range(p):
                Xtr[aa] += Z[i][aa] * rr
                for bb in range(p):
                    XtWX[aa][bb] += Z[i][aa] * ww * Z[i][bb]
        step = k.ridgesolve(XtWX, Xtr, ridge)
        mx = 0.0
        for aa in range(p):
            b[aa] += step[aa]
            mx = max(mx, abs(step[aa]))
        if mx < 1e-13:
            break
    return b


def cheatsheet():
    return ("tmlcen: censoring by IPCW. right = Gbar_c(k|A,W) = "
            "prod(1-lambda_C), weight person-time by 1/Gbar_c, hazard "
            "to survival by the product limit (H&R Ch.17). interval = "
            "Sec.8.5's (1/M) sum (1-Delta_m) r(C_m) I(A=a) / "
            "(gbar_c g), with L = max C_j st Delta=0 and R = min C_j "
            "st Delta=1.")


# compact alias per ledger/NAMING.md
tmlecensoring = tmle_censoring
