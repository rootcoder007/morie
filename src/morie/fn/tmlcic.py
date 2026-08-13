# morie.fn -- function file (rootcoder007/morie)
r"""Adaptive pre-specification: cluster-randomized trial inference.

In a cluster randomized trial the unit is the community, clinic or
school, so there are often only a few dozen independent units and a
great many baseline covariates that might be worth adjusting for.
Adjusting well buys precision; adjusting badly overfits and inflates the
type I error; and choosing after seeing the results is the "fishing
expedition" the pre-specification requirement exists to prevent.

**Adaptive pre-specification** resolves the three at once: pre-specify a
*library* of candidate working models and a *rule* for choosing among
them, and let the data pick. The rule follows empirical efficiency
maximization -- the loss is the squared influence curve of the TMLE, so
its risk is that estimator's asymptotic variance, and the selected
candidate is the one with the smallest cross-validated variance.

**The loss must match the design, and this is the trap the chapter
spells out.** Suppose communities were paired within region because
incidence varies by region. Region is strongly predictive of the
outcome, so the *unmatched* loss -- the sum of squared residuals -- will
happily select the region-adjusted model. That buys nothing: region was
already controlled for in the design. The paired losses

.. math::
   \bar L^P(g_0, \bar Q)(\bar O_j)
     &= \tfrac12 D^P(O_{j1})^2 + \tfrac12 D^P(O_{j2})^2
        - 2\,(Y_{j1} - \bar Q_{j1})(Y_{j2} - \bar Q_{j2}),\\
   \bar L^S(g_0, \bar Q)(\bar O_j) &= \bar D^S(\bar O_j)^2

subtract exactly the within-pair covariance that the design already
bought, so a covariate matched on perfectly earns no credit. The anchor
builds that case and checks the two losses disagree.

**Two targets, two influence curves.** For the population effect
:math:`E[Y_1 - Y_0]`,

.. math:: D^P = \frac{I(A=1)}{g(1|W)} - \frac{I(A=0)}{g(0|W)}
                \big(Y - \bar Q^*(A,W)\big)
                + \bar Q^*(1,W) - \bar Q^*(0,W) - \psi,

while for the sample effect :math:`\frac1n \sum (Y_{1,i} - Y_{0,i})` the
true influence curve involves the counterfactuals, so the plug-in drops
them and keeps only the residual term :math:`D^S`. The sample effect
therefore carries no contribution from the covariate distribution and is
usually estimated more precisely -- conservatively so.

**Estimating a known g.** The allocation probability is 0.5 by design,
and estimating it anyway can still buy precision. The candidate g's are
selected collaboratively, by the same variance loss, so a covariate that
predicts assignment but not the outcome is not adjusted for.

References
----------
Balzer, L. B., van der Laan, M. J. & Petersen, M. L. (2018)
"Data-Adaptive Estimation in Cluster Randomized Trials", Ch. 13 in
van der Laan, M. J. & Rose, S. (eds.) *Targeted Learning in Data
Science: Causal Inference for Complex Longitudinal Studies*, Springer
Series in Statistics, pp. 195-215,
doi:10.1007/978-3-319-65304-4_13. Eq. (13.3), (13.4), (13.5), (13.6),
(13.7), (13.8), (13.9) and Secs. 13.2-13.4. Adapted from Balzer et al.
(2016b).

Balzer, L. B., van der Laan, M. J. & Petersen, M. L. (2016) "Adaptive
pre-specification in randomized trials with and without pair-matching",
*Statistics in Medicine* 35(25), 4528-4545, doi:10.1002/sim.7023. The
paper the chapter is adapted from.

Rubin, D. B. & van der Laan, M. J. (2008) "Empirical Efficiency
Maximization: Improved Locally Efficient Covariate Adjustment in
Randomized Experiments and Survival Analysis", *The International
Journal of Biostatistics* 4(1), article 5, doi:10.2202/1557-4679.1084.
The principle the selector implements.

van der Laan, M. J., Balzer, L. B. & Petersen, M. L. (2013) "Adaptive
Matching in Randomized Trials and Observational Studies", *Journal of
Statistical Research* 46(2), 113-156. Source of the within-pair residual
correlation in eq. (13.7).

Moore, K. L. & van der Laan, M. J. (2009) "Covariate Adjustment in
Randomized Trials with Binary Outcomes: Targeted Maximum Likelihood
Estimation", *Statistics in Medicine* 28(1), 39-64,
doi:10.1002/sim.3445. Why the logistic working model may be
misspecified without cost here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmle_cluster_ic", "adaptive_prespecification",
           "candidate_tmle", "influence_curve", "variance_estimate",
           "default_library"]

_TARGETS = ("SATE", "PATE")
_DESIGNS = ("unmatched", "matched", "clustered")
_EPS = 1e-9


def _logit(p):
    q = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(q / (1.0 - q))


def default_library(p, interactions=True):
    """The chapter's example library: the unadjusted model, one main
    term per covariate, and optionally one treatment interaction each.

    The first candidate has no covariates at all, so the selector can
    always fall back on the unadjusted estimator -- which is the point
    of pre-specifying a library rather than a single model.
    """
    lib = [{"name": "unadjusted", "cols": (), "interact": False}]
    for j in range(p):
        lib.append({"name": "W%d" % (j + 1), "cols": (j,),
                    "interact": False})
    if interactions:
        for j in range(p):
            lib.append({"name": "W%d x A" % (j + 1), "cols": (j,),
                        "interact": True})
    return lib


def _fit_working_model(y, A, W, cand, rows, ridge):
    """logit[Qbar(A,W)] on the candidate's terms, fitted on `rows`.

    Logistic rather than linear because the outcome is bounded, which
    the chapter notes also gives stability under near-positivity
    violations (its footnote 1, citing Gruber & van der Laan 2010b).
    """
    cols = cand["cols"]

    def row(a, i):
        r = [1.0, a] + [W[i][j] for j in cols]
        if cand["interact"]:
            r += [a * W[i][j] for j in cols]
        return r

    X = [row(A[i], i) for i in rows]
    b = k.logit_irls(X, [y[i] for i in rows], ridge=max(ridge, 1e-10))

    def q(a, i):
        r = row(a, i)
        return k.sigmoid(sum(b[t] * r[t] for t in range(len(b))))

    return q, b


def _fit_g(A, W, cand, rows, ridge):
    """A candidate for the exposure mechanism, P(A = 1 | W)."""
    cols = cand["cols"]
    X = [[1.0] + [W[i][j] for j in cols] for i in rows]
    b = k.logit_irls(X, [A[i] for i in rows], ridge=max(ridge, 1e-10))

    def g1(i):
        r = [1.0] + [W[i][j] for j in cols]
        return k.sigmoid(sum(b[t] * r[t] for t in range(len(b))))

    return g1, b


def candidate_tmle(y, A, W, cand, g1, rows=None, eval_rows=None,
                   ridge=1e-8, target_step=True):
    r"""One candidate TMLE: initial fit, targeting, predictions.

    Returns ``(q1, q0, qa, eps)`` evaluated on ``eval_rows``: the
    targeted predictions under treatment and control and at the
    observed exposure.

    The chapter notes the update can be skipped when g is known and the
    working model carries an intercept and a main term for A, because
    the initial fit already solves the score. It is done anyway --
    collaborative g estimation breaks that condition, and when the
    condition does hold the fluctuation returns epsilon of order 1e-16
    rather than something wrong.
    """
    n = len(y)
    rows = list(range(n)) if rows is None else list(rows)
    eval_rows = list(range(n)) if eval_rows is None else list(eval_rows)
    q, _ = _fit_working_model(y, A, W, cand, rows, ridge)

    gA = [0.0] * n
    H = [0.0] * n
    for i in range(n):
        p1 = min(max(g1(i), _EPS), 1.0 - _EPS)
        gA[i] = p1 if A[i] == 1.0 else 1.0 - p1
        H[i] = (1.0 / p1 if A[i] == 1.0 else -1.0 / (1.0 - p1))

    eps = 0.0
    if target_step:
        off = [_logit(q(A[i], i)) for i in range(n)]
        eps = k.logistic_fluctuation(y, off, H, rows)

    q1 = {}
    q0 = {}
    qa = {}
    for i in eval_rows:
        p1 = min(max(g1(i), _EPS), 1.0 - _EPS)
        q1[i] = k.sigmoid(_logit(q(1.0, i)) + eps / p1)
        q0[i] = k.sigmoid(_logit(q(0.0, i)) - eps / (1.0 - p1))
        qa[i] = q1[i] if A[i] == 1.0 else q0[i]
    return q1, q0, qa, {"eps": eps, "gA": gA, "H": H}


def influence_curve(y, A, q1, q0, qa, gA, rows, psi, target):
    """Eq. (13.3) for the PATE and eq. (13.4) for the SATE."""
    if target not in _TARGETS:
        raise ValueError("tmlcic: target must be SATE or PATE, got %r"
                         % (target,))
    out = {}
    for i in rows:
        sign = 1.0 if A[i] == 1.0 else -1.0
        resid = sign * (y[i] - qa[i]) / gA[i]
        if target == "SATE":
            out[i] = resid
        else:
            out[i] = resid + (q1[i] - q0[i]) - psi
    return out


def _pairs_from(cluster, n):
    """Group row indices by pair (or cluster) label, in first-seen order."""
    if cluster is None:
        raise ValueError("tmlcic: a matched or clustered design needs "
                         "the pair labels")
    lab = list(cluster)
    if len(lab) != n:
        raise ValueError("tmlcic: %d pair labels for %d observations"
                         % (len(lab), n))
    order, groups = [], {}
    for i, c in enumerate(lab):
        key = str(c)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    return [groups[c] for c in order]


def variance_estimate(D, y, qa, groups, n, design, target):
    r"""The design's variance estimator.

    ``unmatched``  sample variance of D over the n units, divided by n.
    ``matched``    for the PATE, the same minus twice the within-pair
                   residual correlation of eq. (13.7); for the SATE, the
                   variance of the pair-averaged influence curve over
                   the n/2 pairs.
    ``clustered``  the pair construction with clusters of any size.
    """
    if design == "unmatched":
        v = sum(D[i] * D[i] for i in range(n)) / n
        return v / n, {"unit": "observation", "m": n}
    if target == "PATE":
        # eq. (13.7): the within-pair residual correlation, and the
        # variance it removes
        res = [y[i] - qa[i] for i in range(n)]
        rho = 0.0
        for grp in groups:
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    rho += res[grp[a]] * res[grp[b]]
        rho /= float(len(groups))
        v = sum(D[i] * D[i] for i in range(n)) / n - 2.0 * rho
        return max(v, 0.0) / n, {"unit": "pair", "m": len(groups),
                                 "rho": rho}
    dbar = [sum(D[i] for i in grp) / len(grp) for grp in groups]
    m = len(groups)
    v = sum(x * x for x in dbar) / m
    return v / m, {"unit": "pair", "m": m}


def _loss(D, y, qa, groups, design, target, rows):
    """Eq. (13.5)/(13.6) unmatched, (13.8)/(13.9) matched."""
    if design == "unmatched":
        return sum(D[i] * D[i] for i in rows) / len(rows)
    res = {i: y[i] - qa[i] for i in rows}
    tot, m = 0.0, 0
    for grp in groups:
        g = [i for i in grp if i in res]
        if not g:
            continue
        m += 1
        if target == "PATE":
            val = sum(D[i] * D[i] for i in g) / len(g)
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    val -= 2.0 * res[g[a]] * res[g[b]]
            tot += val
        else:
            dbar = sum(D[i] for i in g) / len(g)
            tot += dbar * dbar
    return tot / m if m else float("inf")


def _cv_folds(groups, n_folds, design, n):
    """Folds that respect the pairing: a pair is never split."""
    if design == "unmatched":
        units = [[i] for i in range(n)]
    else:
        units = groups
    V = len(units) if n_folds in (None, 0, "loo") else \
        max(2, min(int(n_folds), len(units)))
    folds = [[] for _ in range(V)]
    for j, u in enumerate(units):
        folds[j % V].extend(u)
    return [f for f in folds if f]


def adaptive_prespecification(y, A, W, groups, design, target,
                              library=None, g_library=None,
                              n_folds=None, ridge=1e-8):
    r"""Sec. 13.2-13.4: choose the working model, then choose g.

    The Q library is selected first with g held at its known value, then
    the g library is selected collaboratively given the chosen Q -- the
    order the chapter uses, and the reason a covariate already in Q
    earns nothing by also entering g.
    """
    n = len(y)
    p = len(W[0]) if W and W[0] else 0
    lib = default_library(p) if library is None else list(library)
    folds = _cv_folds(groups, n_folds, design, n)

    def known_g(_i):
        return 0.5

    def cv_risk(cand, gfit):
        tot = 0.0
        for val in folds:
            train = [i for i in range(n) if i not in set(val)]
            if not train:
                continue
            g1 = gfit(train)
            q1, q0, qa, info = candidate_tmle(y, A, W, cand, g1,
                                              rows=train, ridge=ridge)
            psi = sum(q1[i] - q0[i] for i in val) / len(val)
            D = influence_curve(y, A, q1, q0, qa, info["gA"], val, psi,
                                target)
            tot += _loss(D, y, qa, groups, design, target, val)
        return tot / len(folds)

    q_risks = [cv_risk(c, lambda _t: known_g) for c in lib]
    best_q = min(range(len(lib)), key=lambda t: q_risks[t])
    chosen = lib[best_q]

    glib = ([{"name": "known (0.5)", "cols": (), "interact": False}]
            + [{"name": "W%d" % (j + 1), "cols": (j,), "interact": False}
               for j in range(p)]) if g_library is None else list(g_library)

    def gfit_for(cand_g):
        if cand_g.get("name") == "known (0.5)" and not cand_g["cols"]:
            return lambda _t: known_g
        return lambda train: _fit_g(A, W, cand_g, train, ridge)[0]

    g_risks = [cv_risk(chosen, gfit_for(cg)) for cg in glib]
    best_g = min(range(len(glib)), key=lambda t: g_risks[t])

    return {"q_candidate": chosen, "q_risks": q_risks,
            "q_names": [c["name"] for c in lib],
            "g_candidate": glib[best_g], "g_risks": g_risks,
            "g_names": [c["name"] for c in glib],
            "gfit": gfit_for(glib[best_g]), "n_folds": len(folds)}


def tmle_cluster_ic(y, D, X, cluster=None, target="SATE",
                    design=None, library=None, g_library=None,
                    n_folds=None, adapt=True, ridge=1e-8, level=0.95):
    r"""TMLE for a cluster randomized trial, with adaptive
    pre-specification of the adjustment set.

    Parameters
    ----------
    y : array-like
        Cluster-level outcome, rescaled to [0, 1] internally; results
        are reported on the original scale.
    D : array-like
        Binary randomization indicator, one per cluster.
    X : array-like
        Cluster-level baseline covariates, n rows.
    cluster : array-like, optional
        Matched-pair labels (or cluster labels for
        ``design="clustered"``). Required unless the design is
        ``"unmatched"``; supplying it selects ``"matched"`` by default.
    target : {"SATE", "PATE"}
        The sample or the population average treatment effect. The
        sample effect is the default: its influence curve drops the
        covariate-distribution term, so it is usually the more precise
        -- conservatively so.
    design : {"unmatched", "matched", "clustered"}, optional
        Inferred from ``cluster`` when omitted.
    library, g_library : sequence of dict, optional
        Candidates as ``{"name", "cols", "interact"}``. Defaults to the
        chapter's example shape: unadjusted, each main term, each
        treatment interaction.
    n_folds : int, optional
        Cross-validation folds; leave-one-out (or leave-one-pair-out)
        by default, which the chapter recommends at these sample sizes.
    adapt : bool
        ``False`` runs the unadjusted TMLE, for comparison.

    Returns
    -------
    RichResult
        ``estimate`` with ``se``, ``ci``, the selected working model and
        exposure model, every candidate's cross-validated risk, and the
        unadjusted comparator.

    Examples
    --------
    A pair-matched trial in 30 communities::

        r = tmle_cluster_ic(y, A, W, cluster=pair, target="SATE")
        r["q_selected"], r["se"], r["se_unadjusted"]
    """
    if target not in _TARGETS:
        raise ValueError("tmlcic: target must be SATE or PATE, got %r"
                         % (target,))
    yv, Av = k.vec(y), k.vec(D)
    n = len(yv)
    if len(Av) != n:
        raise ValueError("tmlcic: %d outcomes but %d treatments"
                         % (n, len(Av)))
    if any(v not in (0.0, 1.0) for v in Av):
        raise ValueError("tmlcic: the randomization indicator must be "
                         "binary 0/1")
    if not 0 < sum(Av) < n:
        raise ValueError("tmlcic: both arms must be non-empty")
    Wm = k.mat(X) if X is not None else [[] for _ in range(n)]
    if len(Wm) != n:
        raise ValueError("tmlcic: %d covariate rows for %d outcomes"
                         % (len(Wm), n))
    if design is None:
        design = "unmatched" if cluster is None else "matched"
    if design not in _DESIGNS:
        raise ValueError("tmlcic: design must be one of %s, got %r"
                         % (", ".join(_DESIGNS), design))
    groups = (_pairs_from(cluster, n) if design != "unmatched"
              else [[i] for i in range(n)])
    if design == "matched" and any(len(g) != 2 for g in groups):
        raise ValueError("tmlcic: design='matched' needs pairs; use "
                         "design='clustered' for other sizes")
    if n < 4:
        raise ValueError("tmlcic: need at least 4 units, got %d" % n)

    ymin, ymax = min(yv), max(yv)
    rng = ymax - ymin
    if rng <= 0.0:
        raise ValueError("tmlcic: the outcome is constant")
    ys = [(v - ymin) / rng for v in yv]

    unadj = {"name": "unadjusted", "cols": (), "interact": False}

    def known_g(_i):
        return 0.5

    if adapt:
        sel = adaptive_prespecification(ys, Av, Wm, groups, design,
                                        target, library=library,
                                        g_library=g_library,
                                        n_folds=n_folds, ridge=ridge)
        cand, gfit = sel["q_candidate"], sel["gfit"]
        g1 = gfit(list(range(n)))
    else:
        sel = {"q_candidate": unadj, "q_risks": [], "q_names": [],
               "g_candidate": {"name": "known (0.5)"}, "g_risks": [],
               "g_names": [], "n_folds": 0}
        cand, g1 = unadj, known_g

    q1, q0, qa, info = candidate_tmle(ys, Av, Wm, cand, g1, ridge=ridge)
    rows = list(range(n))
    psi_s = sum(q1[i] - q0[i] for i in rows) / n
    Dic = influence_curve(ys, Av, q1, q0, qa, info["gA"], rows, psi_s,
                          target)
    var_s, vinfo = variance_estimate(Dic, ys, qa, groups, n, design,
                                     target)

    # the unadjusted comparator, same design and target
    u1, u0, ua, uinfo = candidate_tmle(ys, Av, Wm, unadj, known_g,
                                       ridge=ridge)
    psi_u = sum(u1[i] - u0[i] for i in rows) / n
    Du = influence_curve(ys, Av, u1, u0, ua, uinfo["gA"], rows, psi_u,
                         target)
    var_u, _ = variance_estimate(Du, ys, ua, groups, n, design, target)

    psi = rng * psi_s
    se = rng * math.sqrt(var_s)
    se_u = rng * math.sqrt(var_u)
    z = k.qnorm(0.5 + 0.5 * float(level))

    return RichResult(payload={
        "estimate": psi, "se": se, "n": n,
        "ci": (psi - z * se, psi + z * se),
        "level": float(level),
        "unadjusted": rng * psi_u, "se_unadjusted": se_u,
        "variance_ratio": (var_s / var_u) if var_u > 0.0
        else float("nan"),
        "q_selected": sel["q_candidate"]["name"],
        "q_risks": dict(zip(sel["q_names"], sel["q_risks"])),
        "g_selected": sel["g_candidate"]["name"],
        "g_risks": dict(zip(sel["g_names"], sel["g_risks"])),
        "epsilon": info["eps"],
        "influence_curve": [Dic[i] * rng for i in rows],
        "eic_mean": sum(Dic.values()) / n,
        "rho": vinfo.get("rho", float("nan")),
        "independent_units": vinfo["m"], "unit": vinfo["unit"],
        "design": design, "target": target,
        "n_folds": sel["n_folds"], "adapt": bool(adapt),
        "method": "adaptive pre-specification TMLE, Balzer, van der "
                  "Laan & Petersen (2018) Ch. 13",
    })


def cheatsheet():
    return ("tmlcic: cluster randomized trial. Pre-specify a LIBRARY of "
            "working models, select by cross-validated squared "
            "influence curve (the TMLE's own variance). Losses: 13.5 "
            "PATE / 13.6 SATE unmatched, 13.8 / 13.9 matched -- the "
            "matched ones subtract the within-pair residual covariance "
            "so a perfectly matched covariate earns no credit. Then "
            "select g collaboratively by the same loss. Pairs are never "
            "split across folds.")


# compact alias per ledger/NAMING.md
tmleclusteric = tmle_cluster_ic
