# morie.fn -- function file (rootcoder007/morie)
r"""Collaborative targeted minimum loss-based estimation (C-TMLE).

The word doing the work is *collaborative*. An ordinary TMLE fits the
treatment mechanism :math:`\bar G` as well as it can, then targets the
outcome fit with it. C-TMLE instead builds a **sequence** of treatment
mechanisms and picks among them by how well the resulting *targeted
outcome* fits -- not by how well :math:`\bar G` predicts treatment.

The consequence, and the reason the method exists, is stated in the
chapter: it "makes sure that instrumental variables will only be
included in the fit of the treatment mechanism at large enough sample
sizes for which the parametric maximum likelihood estimator extension
in the update step using this covariate results in a statistically
significant gain in fit". An instrument -- a covariate that predicts
treatment strongly and the outcome not at all -- inflates the weights
and the variance while removing no confounding. Selecting
:math:`\bar G` by its own likelihood puts the instrument in. Selecting
by the outcome fit leaves it out. The anchor plants exactly such an
instrument and checks which happens.

Example 10.3 of the chapter gives the machinery. The least favourable
submodel through the initial outcome fit is

.. math:: \mathrm{logit}\,\bar Q_{n,\epsilon,G_{n,h}}
          = \mathrm{logit}\,\bar Q_n + \epsilon\, C(\bar G_{n,h}),
          \qquad C(\bar G)(A, W) = \frac{A}{\bar G(W)},

with :math:`\epsilon` the maximum likelihood estimate under the
log-likelihood loss
:math:`L(\bar Q) = -\{Y\log\bar Q + (1-Y)\log(1-\bar Q)\}`, and the
resulting estimator solves
:math:`P_n D_1^*(\bar Q^*_{n,h}, G_{n,h}) = 0` with
:math:`D_1^*(\bar Q,\bar G) = \frac{A}{\bar G}(Y - \bar Q)`.

Two tuning regimes, both implemented, because the chapter is explicitly
about the difference between them:

``"discrete"``
    The classical C-TMLE of van der Laan & Gruber (2010): the sequence
    is built by adding covariates to the treatment model one at a time,
    greedily, choosing at each step the covariate whose inclusion most
    improves the targeted outcome fit.

``"continuous"``
    This chapter's subject: the sequence is indexed by a continuous
    tuning parameter of the treatment-model fit -- here the ridge
    penalty, the chapter's example being "the L1-penalty in a lasso
    regression of the treatment mechanism (or a bandwidth of a kernel
    regression smoother)". The chapter's point is that with a discrete
    sequence any reasonable selector ends up at the most nonparametric
    estimator asymptotically, so C-TMLE and TMLE agree; with a
    continuous one they need not.

**Nesting is required, not cosmetic.** The chapter builds the sequence
so "the empirical fits are increasing as h approximates 0", with
:math:`\bar Q^*_{n,h}` using a previous :math:`\bar Q_{n,h'}`,
:math:`h' > h`, as its initial estimator. Without nesting the fits are
not comparable across h and the selection step is meaningless. Each
step here therefore starts from the previous step's targeted fit.

References
----------
van der Laan, M. J. & Rose, S. (eds.) (2018) *Targeted Learning in
Data Science: Causal Inference for Complex Longitudinal Studies*,
Springer Series in Statistics, doi:10.1007/978-3-319-65304-4, Ch. 10
"C-TMLE for Continuous Tuning" -- Sec. 10.1, Examples 10.2 and 10.3,
and Sec. 10.1.1 on discrete versus continuous tuning.

van der Laan, M. J. & Gruber, S. (2010) "Collaborative double robust
targeted maximum likelihood estimation", *The International Journal of
Biostatistics* 6(1), article 17, doi:10.2202/1557-4679.1181 -- the
original, discrete-tuning C-TMLE.

Gruber, S. & van der Laan, M. J. (2010) "A targeted maximum likelihood
estimator of a causal effect on a bounded continuous outcome", *The
International Journal of Biostatistics* 6(1), article 26,
doi:10.2202/1557-4679.1260 -- the scaling that lets the logistic
fluctuation carry a continuous outcome.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmle_cdrs", "ctmle_sequence"]

_TUNING = ("discrete", "continuous")


def _fluctuate(qa, q1, q0, y, d, g, clip=1e-8):
    """One targeting step: the Example 10.3 submodel, solved by Newton.

    Returns the updated (qa, q1, q0) and the fitted epsilon. The clever
    covariate is A/G for the treated arm and -(1-A)/(1-G) for the
    control arm, which is the two-armed form of C(G) whose score gives
    the ATE component of the efficient influence curve.
    """
    n = len(y)
    h = [d[i] / g[i] - (1.0 - d[i]) / (1.0 - g[i]) for i in range(n)]
    eps = 0.0
    for _ in range(60):
        num = den = 0.0
        for i in range(n):
            z = math.log(qa[i] / (1.0 - qa[i])) + eps * h[i]
            p = k.sigmoid(z)
            num += h[i] * (y[i] - p)
            den += h[i] * h[i] * p * (1.0 - p)
        if den <= 0.0:
            break
        step = num / den
        eps += step
        if abs(step) < 1e-13:
            break
    qa2, q1b, q0b = [], [], []
    for i in range(n):
        qa2.append(min(max(k.sigmoid(math.log(qa[i] / (1.0 - qa[i]))
                                     + eps * h[i]), clip), 1.0 - clip))
        q1b.append(min(max(k.sigmoid(math.log(q1[i] / (1.0 - q1[i]))
                                     + eps / g[i]), clip), 1.0 - clip))
        q0b.append(min(max(k.sigmoid(math.log(q0[i] / (1.0 - q0[i]))
                                     - eps / (1.0 - g[i])), clip),
                       1.0 - clip))
    return qa2, q1b, q0b, eps


def _q_loss(qa, y):
    """L(Qbar) = -{Y log Qbar + (1-Y) log(1-Qbar)}, the chapter's loss."""
    tot = 0.0
    for i in range(len(y)):
        tot -= (y[i] * math.log(qa[i])
                + (1.0 - y[i]) * math.log(1.0 - qa[i]))
    return tot / len(y)


def _propensity(d, cols, n, penalty=0.0, trim=0.005):
    Z = k.design([[c[i] for c in cols] for i in range(n)] if cols
                 else None, n)
    b = k.logit_irls(Z, d, 60, 1e-10, penalty=float(penalty))
    g = [min(max(k.sigmoid(v), trim), 1.0 - trim)
         for v in k.matvec(Z, b)]
    return g, b


def ctmle_sequence(y, D, X, tuning="discrete", penalties=None,
                   trim=0.005, scale=None, q_covariates=None):
    """Build the nested sequence of (G, targeted Q) and score each.

    Returns one record per step with the treatment model used, the
    targeted outcome loss, and the resulting estimate. The selection
    itself is done by :func:`tmle_cdrs`.
    """
    if tuning not in _TUNING:
        raise ValueError("ctmle_sequence: tuning must be 'discrete' or "
                         "'continuous', got %r" % (tuning,))
    yv, d = k.vec(y), k.vec(D)
    n = len(yv)
    if len(d) != n:
        raise ValueError("ctmle_sequence: %d outcomes but %d treatments"
                         % (n, len(d)))
    if any(v not in (0.0, 1.0) for v in d):
        raise ValueError("ctmle_sequence: treatment must be binary 0/1")
    Xm = k.mat(X) if X is not None else [[] for _ in range(n)]
    p = len(Xm[0]) if Xm and Xm[0] else 0
    cols = [[Xm[i][j] for i in range(n)] for j in range(p)]

    # Gruber & van der Laan (2010): scale Y into [0,1] so the logistic
    # fluctuation is valid for a continuous outcome too.
    lo, hi = min(yv), max(yv)
    rng = (hi - lo) if hi > lo else 1.0
    if scale is None:
        scale = not all(v in (0.0, 1.0) for v in yv)
    ys = [(v - lo) / rng for v in yv] if scale else list(yv)
    ys = [min(max(v, 1e-8), 1.0 - 1e-8) for v in ys]

    # Initial outcome fit Qbar_n. WHICH covariates it sees matters more
    # than it looks: fitted by least squares on the same covariates that
    # build G, its residual is orthogonal to the clever covariate by
    # construction, epsilon comes out at zero and the targeting step is
    # a no-op -- so every candidate G scores identically and the
    # collaborative selection has nothing to select on. The chapter is
    # explicitly about the case where "the initial estimator is
    # inconsistent", so q_covariates lets the caller say what the
    # outcome model actually saw. None means all of them.
    qcols = (list(range(p)) if q_covariates is None
             else [int(c) for c in q_covariates])
    for c in qcols:
        if not 0 <= c < p:
            raise ValueError(
                "ctmle_sequence: q_covariates index %d is outside the "
                "%d covariates supplied" % (c, p))
    Zq = k.design([[d[i]] + [Xm[i][c] for c in qcols]
                   for i in range(n)], n)
    bq = k.lstsq(Zq, ys)

    def q_at(a, i):
        row = [1.0, a] + [Xm[i][c] for c in qcols]
        return min(max(sum(bq[j] * row[j] for j in range(len(bq))),
                       1e-8), 1.0 - 1e-8)
    qa = [q_at(d[i], i) for i in range(n)]
    q1 = [q_at(1.0, i) for i in range(n)]
    q0 = [q_at(0.0, i) for i in range(n)]

    steps = []
    if tuning == "discrete":
        chosen, remaining = [], list(range(p))
        # step 0: intercept-only treatment model
        g, _ = _propensity(d, [], n, trim=trim)
        qa, q1, q0, eps = _fluctuate(qa, q1, q0, ys, d, g)
        steps.append({"step": 0, "covariates": [], "loss": _q_loss(qa, ys),
                      "epsilon": eps, "g": g,
                      "psi": rng * sum(q1[i] - q0[i]
                                       for i in range(n)) / n})
        while remaining:
            best = None
            for j in remaining:
                gj, _ = _propensity(d, [cols[c] for c in chosen + [j]],
                                    n, trim=trim)
                # nested: start from the CURRENT targeted fit
                a2, b2, c2, e2 = _fluctuate(qa, q1, q0, ys, d, gj)
                loss = _q_loss(a2, ys)
                if best is None or loss < best[0]:
                    best = (loss, j, gj, a2, b2, c2, e2)
            loss, j, gj, a2, b2, c2, e2 = best
            chosen.append(j)
            remaining.remove(j)
            qa, q1, q0 = a2, b2, c2
            steps.append({"step": len(chosen), "covariates": list(chosen),
                          "loss": loss, "epsilon": e2, "g": gj,
                          "psi": rng * sum(q1[i] - q0[i]
                                           for i in range(n)) / n})
    else:
        if penalties is None:
            penalties = [1e4, 1e3, 1e2, 10.0, 1.0, 0.1, 1e-2, 0.0]
        for s, lam in enumerate(penalties):
            g, _ = _propensity(d, cols, n, penalty=float(lam), trim=trim)
            qa, q1, q0, eps = _fluctuate(qa, q1, q0, ys, d, g)
            steps.append({"step": s, "penalty": float(lam),
                          "loss": _q_loss(qa, ys), "epsilon": eps,
                          "g": g,
                          "psi": rng * sum(q1[i] - q0[i]
                                           for i in range(n)) / n})
    return steps, {"scale": rng, "shift": lo, "n": n, "p": p,
                   "y_scaled": ys, "treatment": d, "columns": cols,
                   "q_covariates": qcols}


def tmle_cdrs(y, D, X, tuning="discrete", penalties=None, n_folds=5,
              trim=0.005, scale=None, q_covariates=None):
    r"""Collaborative TMLE for the ATE.

    Parameters
    ----------
    y, D, X : array-like
        Outcome, binary treatment, covariates.
    tuning : {"discrete", "continuous"}
        How the sequence of treatment mechanisms is indexed.
    penalties : list of float, optional
        The continuous tuning path, largest penalty first so the fits
        are nested and increasing.
    n_folds : int
        Folds for the cross-validated selection of the step.

    Returns
    -------
    RichResult
        ``estimate`` is the ATE at the selected step, with the whole
        sequence in ``steps`` and the selected index in ``selected``.

    Examples
    --------
    An instrument in the covariates is not selected into the treatment
    model, which is the point of the method::

        r = tmle_cdrs(y, D, X)
        r["selected_covariates"]
    """
    steps, info = ctmle_sequence(y, D, X, tuning=tuning,
                                 penalties=penalties, trim=trim,
                                 scale=scale, q_covariates=q_covariates)
    n = info["n"]
    folds = [[i for i in range(n) if i % int(n_folds) == f]
             for f in range(int(n_folds))]
    # Cross-validated version of the same loss: the chapter selects h by
    # "the L-fit of Q*_{n,h}", and the honest version of that fit is
    # out-of-sample.
    cv = []
    for s in range(len(steps)):
        tot = 0.0
        for f in folds:
            tr = [i for i in range(n) if i not in set(f)]
            sub = _refit_on(info, steps, s, tr, f, tuning, penalties,
                            trim)
            tot += sub
        cv.append(tot / len(folds))
    sel = min(range(len(steps)), key=lambda s: cv[s])
    best = steps[sel]

    return RichResult(payload={
        "estimate": best["psi"],
        "psi": best["psi"],
        "selected": sel,
        "selected_covariates": best.get("covariates"),
        "selected_penalty": best.get("penalty"),
        "steps": [{kk: vv for kk, vv in st.items() if kk != "g"}
                  for st in steps],
        "cv_loss": cv,
        "in_sample_loss": [st["loss"] for st in steps],
        "epsilon": best["epsilon"],
        "tuning": tuning, "n": n, "n_covariates": info["p"],
        "method": "collaborative TMLE, van der Laan & Rose (2018) "
                  "Ch. 10 Example 10.3 with %s tuning" % tuning,
    })


def _refit_on(info, steps, s, tr, fold, tuning, penalties, trim):
    """Out-of-sample loss of step s, refitting on the training rows."""
    ys, d, cols = info["y_scaled"], info["treatment"], info["columns"]
    ntr = len(tr)
    if ntr < 5:
        return float("inf")
    st = steps[s]
    use = st.get("covariates")
    lam = st.get("penalty", 0.0)
    sub_cols = ([[cols[c][i] for i in tr] for c in use] if use is not None
                else [[c[i] for i in tr] for c in cols])
    dtr = [d[i] for i in tr]
    ytr = [ys[i] for i in tr]
    if len(set(dtr)) < 2:
        return float("inf")
    g_tr, bg = _propensity(dtr, sub_cols, ntr, penalty=lam, trim=trim)
    # initial Q on the training rows
    qc = info["q_covariates"]
    Xtr = [[cols[c][i] for c in qc] for i in tr]
    Zq = k.design([[dtr[j]] + Xtr[j] for j in range(ntr)], ntr)
    bq = k.lstsq(Zq, ytr)

    def q_at(a, row):
        r = [1.0, a] + list(row)
        return min(max(sum(bq[j] * r[j] for j in range(len(bq))), 1e-8),
                   1.0 - 1e-8)
    qa = [q_at(dtr[j], Xtr[j]) for j in range(ntr)]
    q1 = [q_at(1.0, Xtr[j]) for j in range(ntr)]
    q0 = [q_at(0.0, Xtr[j]) for j in range(ntr)]
    _, _, _, eps = _fluctuate(qa, q1, q0, ytr, dtr, g_tr)

    # evaluate on the held-out rows with the training-fitted pieces
    tot, m = 0.0, 0
    Zg = k.design([[cols[c][i] for c in (use if use is not None
                                         else range(len(cols)))]
                   for i in fold] if (cols and (use is None or use))
                  else None, len(fold))
    gs = [min(max(k.sigmoid(v), trim), 1.0 - trim)
          for v in k.matvec(Zg, bg)] if len(bg) == len(Zg[0]) else None
    for idx, i in enumerate(fold):
        row = [cols[c][i] for c in qc]
        q = q_at(d[i], row)
        gi = gs[idx] if gs else 0.5
        h = d[i] / gi - (1.0 - d[i]) / (1.0 - gi)
        qs = k.sigmoid(math.log(q / (1.0 - q)) + eps * h)
        qs = min(max(qs, 1e-8), 1.0 - 1e-8)
        tot -= (ys[i] * math.log(qs)
                + (1.0 - ys[i]) * math.log(1.0 - qs))
        m += 1
    return tot / m if m else float("inf")


def cheatsheet():
    return ("tmlcds: collaborative TMLE. Build a NESTED sequence of "
            "treatment mechanisms and select by the cross-validated "
            "loss of the TARGETED OUTCOME fit, not of G itself -- so an "
            "instrument stays out of the propensity model. Submodel "
            "logit Q* = logit Q + eps A/G (vdL & Rose 2018 Ch.10 Ex.10.3). "
            "tuning = discrete (greedy covariates) or continuous (penalty).")


# compact alias per ledger/NAMING.md
tmlecdrs = tmle_cdrs
