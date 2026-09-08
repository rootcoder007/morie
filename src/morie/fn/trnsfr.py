# morie.fn -- function file (rootcoder007/morie)
r"""Transporting a treatment effect from one cohort to another.

An effect estimated in one population is a statement about *that*
population. Moving it to another is only licensed when the two differ
in ways you have measured -- and the correction is a reweighting, not a
hope.

**The identity that does the work.** Suppose the source cohort
:math:`S=1` and the target cohort :math:`S=0` share the same
conditional effect function :math:`\tau(x) = E[Y(1)-Y(0)\mid X=x]` but
differ in the distribution of :math:`X`. Then

.. math:: \tau_{\text{target}}
          = E_{P_0}[\tau(X)]
          = E_{P_1}\!\left[\frac{p_0(X)}{p_1(X)}\,\tau(X)\right],

so the source-cohort estimate needs each unit weighted by the density
ratio between target and source. That ratio never has to be modelled
directly: with :math:`\pi(x) = P(S=1 \mid X=x)` estimated from the
pooled data,

.. math:: \frac{p_0(x)}{p_1(x)}
          = \frac{P(S=0)}{P(S=1)}\cdot\frac{1-\pi(x)}{\pi(x)},

which is a logistic regression away. This is the same
inverse-probability logic Wager's Chapter 2 applies to treatment
assignment, applied instead to *cohort membership*.

**Overlap is the binding assumption, and it is checkable.** If some
region of the target's covariate space has no source support, the
weight there is infinite and no amount of reweighting rescues it --
the effect for those units is not identified, full stop. Rather than
letting that show up as a large variance, ``transport_weights``
reports the largest weight and the effective sample size
:math:`(\sum w)^2/\sum w^2`, and refuses membership probabilities at
the boundary.

**Weighting is not the only route, and the alternatives are not
equivalent.** ``method="ipw"`` reweights; ``method="outcome"`` fits
:math:`\hat\tau(x)` in the source and averages it over the target
covariates; ``method="dr"`` combines them and is consistent if either
piece is right. All three are here because the source literature
carries all three, and ``"dr"`` is the default for that reason.

**Balancing weights instead of inverse probabilities.** Chapter 7's
point is that the weights can be chosen to *make the moments match*
rather than by inverting a fitted probability -- solve directly for
non-negative :math:`w` minimising :math:`\sum w_i^2` subject to
:math:`\sum_i w_i X_i = \bar X_{\text{target}}`. When the propensity
model is misspecified, the balancing weights still balance the moments
they were told to balance, and the inverse-probability weights do not.
That is ``method="balance"``.

**Marginal structural models across cohorts.** When the treatment is a
regime rather than a point exposure, the same weights multiply the
IPW weights of the MSM: a unit contributes
:math:`w^{\text{transport}}_i \cdot w^{\text{MSM}}_i`, and the MSM is
then fitted by weighted least squares. That is what
``transfer_msm`` does, and it is why the transported estimate can be
read off the same coefficient table as the source-cohort one.

References
----------
Wager, S. (2025) *Causal Inference: A Statistical Learning Approach*,
Stanford University, draft of 26 November 2025. Chapter 2, Sec. 2.2
(inverse-propensity weighting and the overlap condition); Chapter 3
(the doubly robust/AIPW construction reused here for the ``"dr"``
route); Chapter 7, Sec. 7.1-7.2 (covariate-balancing weights and the
augmented balancing estimator).

Notes
-----
The ledger recorded this module as "Athey-Wager (2019),
transfer-learning MSM across cohorts". No paper of that title was
located and the entry came from the generated stub; the implementation
follows Wager (2025) chapters 2, 3 and 7.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["transport_weights", "balancing_weights", "transport_ate",
           "transfer_msm"]

_EPS = 1e-12
_METHODS = ("dr", "ipw", "outcome", "balance")


def _cohort(S):
    s = [float(v) for v in k.vec(S)]
    for v in s:
        if v not in (0.0, 1.0):
            raise ValueError("trnsfr: the cohort indicator must be "
                             "0/1 (1 = source), got %r" % (v,))
    if sum(s) < 2 or len(s) - sum(s) < 2:
        raise ValueError("trnsfr: both cohorts need at least 2 units "
                         "(source %d, target %d)"
                         % (int(sum(s)), int(len(s) - sum(s))))
    return s


def transport_weights(X, S, trim=1e-3, ridge=1e-6):
    r"""Odds-of-membership weights for the source cohort.

    Fits :math:`\pi(x) = P(S=1\mid X=x)` by logistic regression on the
    pooled data and returns, for each **source** unit,
    :math:`w_i = (1-\pi(X_i))/\pi(X_i)`, normalised to mean 1. Target
    units get weight 0.

    ``trim`` refuses membership probabilities within ``trim`` of 0 or 1
    rather than returning a near-infinite weight: outside the overlap
    region the transported effect is not identified, and a huge weight
    is a symptom of that, not a solution to it.
    """
    Xm = k.mat(X)
    s = _cohort(S)
    n = len(Xm)
    if len(s) != n:
        raise ValueError("trnsfr: %d cohort labels for %d rows"
                         % (len(s), n))
    D = k.design(Xm, n)
    beta = k.logit_irls(D, s, ridge=ridge)
    pi = [k.sigmoid(sum(D[i][j] * beta[j] for j in range(len(beta))))
          for i in range(n)]
    lo, hi = float(trim), 1.0 - float(trim)
    bad = [i for i in range(n) if not lo <= pi[i] <= hi]
    if bad:
        raise ValueError("trnsfr: %d unit(s) have a cohort-membership "
                         "probability outside [%g, %g] (extreme %.4g) "
                         "-- there is no overlap there and the "
                         "transported effect is not identified for "
                         "them" % (len(bad), lo, hi,
                                   min(pi[i] for i in bad)
                                   if pi[bad[0]] < lo
                                   else max(pi[i] for i in bad)))
    raw = [((1.0 - pi[i]) / pi[i]) if s[i] == 1.0 else 0.0
           for i in range(n)]
    tot = sum(raw)
    if tot <= _EPS:
        raise ValueError("trnsfr: the transport weights are all zero")
    ns = int(sum(s))
    w = [v * ns / tot for v in raw]
    ess = (sum(w) ** 2) / sum(v * v for v in w)
    return {"weights": w, "pi": pi, "max_weight": max(w),
            "ess": ess, "ess_fraction": ess / ns, "n_source": ns,
            "coef": beta,
            "method": "odds of cohort membership; Wager (2025) "
                      "Sec. 2.2 applied to S rather than W"}


def balancing_weights(X, S, ridge=1e-8):
    r"""Minimum-variance weights that match the target's X means.

    Solves :math:`\min_w \sum_i w_i^2` over source units subject to
    :math:`\sum_i w_i \tilde X_i = \bar{\tilde X}_{\text{target}}`
    with :math:`\tilde X = (1, X)`, so the first constraint is
    :math:`\sum w_i = 1`. The solution is the minimum-norm one,
    :math:`w = \tilde X_1^{\top}(\tilde X_1 \tilde X_1^{\top})^{-1}
    \bar{\tilde X}_0`.

    The moments named in the constraint are balanced **exactly**, by
    construction, whether or not any model is right -- which is the
    property inverse-probability weights lack. Weights may go negative;
    that is reported rather than hidden, since a negative weight means
    the target mean sits outside the convex hull of the source
    covariates.
    """
    Xm = k.mat(X)
    s = _cohort(S)
    n = len(Xm)
    if len(s) != n:
        raise ValueError("trnsfr: %d cohort labels for %d rows"
                         % (len(s), n))
    D = k.design(Xm, n)
    p = len(D[0])
    src = [i for i in range(n) if s[i] == 1.0]
    tgt = [i for i in range(n) if s[i] == 0.0]
    if len(src) < p:
        raise ValueError("trnsfr: %d source units cannot balance %d "
                         "moments" % (len(src), p))
    b = [sum(D[i][j] for i in tgt) / len(tgt) for j in range(p)]
    # G = A A^T with A the p-by-|src| constraint matrix
    G = [[sum(D[i][a] * D[i][c] for i in src) + (ridge if a == c else 0.0)
          for c in range(p)] for a in range(p)]
    lam = k.cholsolve(G, b)
    w = [0.0] * n
    for i in src:
        w[i] = sum(D[i][a] * lam[a] for a in range(p))
    achieved = [sum(w[i] * D[i][j] for i in src) for j in range(p)]
    err = max(abs(achieved[j] - b[j]) for j in range(p))
    pos = sum(w[i] for i in src if w[i] > 0.0)
    return {"weights": w, "target_moments": b, "achieved": achieved,
            "max_imbalance": err, "n_negative":
            sum(1 for i in src if w[i] < 0.0),
            "positive_mass": pos,
            "method": "minimum-variance covariate balancing weights; "
                      "Wager (2025) Sec. 7.1"}


def transport_ate(Y, W, X, S, method="dr", e=None, trim=1e-3,
                  ridge=1e-6):
    r"""The source-cohort effect, transported to the target cohort.

    ``method``
        ``"ipw"`` weights source units by the membership odds;
        ``"outcome"`` fits :math:`\hat\tau(x)` in the source by a
        linear model with treatment interactions and averages it over
        the **target** covariates; ``"dr"`` combines them so that
        either piece being right suffices; ``"balance"`` uses the
        exactly-balancing weights instead of the fitted ones.
    """
    if method not in _METHODS:
        raise ValueError("trnsfr: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    y = [float(v) for v in k.vec(Y)]
    w = [float(v) for v in k.vec(W)]
    Xm = k.mat(X)
    s = _cohort(S)
    n = len(y)
    for nm, v in (("W", w), ("X", Xm), ("S", s)):
        if len(v) != n:
            raise ValueError("trnsfr: %s has %d rows for %d outcomes"
                             % (nm, len(v), n))
    for v in w:
        if v not in (0.0, 1.0):
            raise ValueError("trnsfr: W must be 0/1, got %r" % (v,))
    src = [i for i in range(n) if s[i] == 1.0]
    tgt = [i for i in range(n) if s[i] == 0.0]
    if not any(w[i] == 1.0 for i in src) or \
            not any(w[i] == 0.0 for i in src):
        raise ValueError("trnsfr: the source cohort must contain both "
                         "treated and control units")
    ps = ([0.5] * n if e is None
          else ([float(e)] * n if isinstance(e, (int, float))
                else [float(v) for v in k.vec(e)]))
    if any(not 0.0 < v < 1.0 for v in ps):
        raise ValueError("trnsfr: the treatment propensity must lie "
                         "strictly in (0, 1)")

    # tau(x) fitted in the source by an interacted linear model
    Dx = k.design(Xm, n)
    p = len(Dx[0])
    rows = [Dx[i] + [w[i] * v for v in Dx[i]] for i in src]
    beta = k.lstsq(rows, [y[i] for i in src], 1e-8)
    def tau_hat(i):
        return sum(Dx[i][j] * beta[p + j] for j in range(p))
    def mu(i, wv):
        return (sum(Dx[i][j] * beta[j] for j in range(p))
                + wv * sum(Dx[i][j] * beta[p + j] for j in range(p)))

    out_part = sum(tau_hat(i) for i in tgt) / len(tgt)
    if method == "outcome":
        est, diag = out_part, {}
    else:
        if method == "balance":
            wd = balancing_weights(Xm, s, ridge=1e-8)
            tw = wd["weights"]
            norm = sum(tw[i] for i in src)
        else:
            wd = transport_weights(Xm, s, trim=trim, ridge=ridge)
            tw = wd["weights"]
            norm = sum(tw[i] for i in src)
        if abs(norm) <= _EPS:
            raise ValueError("trnsfr: the transport weights sum to 0")
        if method == "ipw" or method == "balance":
            # Hajek (self-normalised) form: each arm is divided by the
            # weight it actually received, not by the common total.
            # The Horvitz-Thompson version -- one shared denominator --
            # is unbiased only if the realised arm weights happen to
            # match their expectations, and the sampling error in that
            # match is a first-order bias at these sample sizes.
            n1 = sum(tw[i] * w[i] / ps[i] for i in src)
            n0 = sum(tw[i] * (1.0 - w[i]) / (1.0 - ps[i])
                     for i in src)
            if abs(n1) <= _EPS or abs(n0) <= _EPS:
                raise ValueError("trnsfr: one treatment arm carries no "
                                 "transport weight (treated %.3g, "
                                 "control %.3g)" % (n1, n0))
            est = (sum(tw[i] * w[i] * y[i] / ps[i] for i in src) / n1
                   - sum(tw[i] * (1.0 - w[i]) * y[i] / (1.0 - ps[i])
                         for i in src) / n0)
        else:                                   # dr
            num = sum(tw[i] * (mu(i, 1.0) - mu(i, 0.0)
                               + w[i] * (y[i] - mu(i, 1.0)) / ps[i]
                               - (1.0 - w[i]) * (y[i] - mu(i, 0.0))
                               / (1.0 - ps[i]))
                      for i in src)
            est = num / norm
        diag = {kk: wd[kk] for kk in wd if kk != "weights"}
    naive = (sum(y[i] * w[i] for i in src)
             / max(sum(w[i] for i in src), _EPS)
             - sum(y[i] * (1.0 - w[i]) for i in src)
             / max(sum(1.0 - w[i] for i in src), _EPS))
    return RichResult(payload={
        "estimate": est, "source_ate": naive,
        "outcome_route": out_part,
        "n_source": len(src), "n_target": len(tgt),
        "method": method, "diagnostics": diag,
        "assumption": "the conditional effect function is shared "
                      "across cohorts and the target's covariate "
                      "support lies inside the source's",
    })


def transfer_msm(Y, A, H, cohort, target=0, e=None, trim=1e-3,
                 ridge=1e-6):
    r"""A marginal structural model fitted with transported weights.

    ``A`` is the exposure, ``H`` the history/covariates used both to
    build the IPW weights and to define the cohorts, and ``cohort`` the
    cohort label. Every unit contributes
    :math:`w^{\text{transport}}_i \cdot w^{\text{MSM}}_i`, so the MSM
    coefficient reported is the one that would have been obtained had
    the source cohort had the target's covariate distribution.
    """
    y = [float(v) for v in k.vec(Y)]
    a = [float(v) for v in k.vec(A)]
    Hm = k.mat(H)
    lab = [str(c) for c in cohort]
    n = len(y)
    if not (len(a) == len(Hm) == len(lab) == n):
        raise ValueError("trnsfr: Y, A, H and cohort must agree in "
                         "length (%d, %d, %d, %d)"
                         % (n, len(a), len(Hm), len(lab)))
    tgt = str(target)
    if tgt not in set(lab):
        raise ValueError("trnsfr: target cohort %r is not present; "
                         "cohorts are %s"
                         % (target, sorted(set(lab))))
    S = [0.0 if c == tgt else 1.0 for c in lab]
    tw = transport_weights(Hm, S, trim=trim, ridge=ridge)["weights"]
    if e is None:
        Dh = k.design(Hm, n)
        bh = k.logit_irls(Dh, [1.0 if v > 0.0 else 0.0 for v in a],
                          ridge=ridge)
        ps = [k.sigmoid(sum(Dh[i][j] * bh[j] for j in range(len(bh))))
              for i in range(n)]
    else:
        ps = ([float(e)] * n if isinstance(e, (int, float))
              else [float(v) for v in k.vec(e)])
    if any(not 0.0 < v < 1.0 for v in ps):
        raise ValueError("trnsfr: the exposure propensity must lie "
                         "strictly in (0, 1)")
    msm_w = [(1.0 / ps[i]) if a[i] > 0.0 else (1.0 / (1.0 - ps[i]))
             for i in range(n)]
    tot = [tw[i] * msm_w[i] for i in range(n)]
    # Fit on the SOURCE cohort only. The whole point is that the target
    # supplies covariates, not outcomes; pooling the two would return a
    # mixture of the source and target effects rather than the target's.
    rows = [[a[i]] for i in range(n) if S[i] == 1.0]
    ys = [y[i] for i in range(n) if S[i] == 1.0]
    ws = [tot[i] for i in range(n) if S[i] == 1.0]
    if len({r[0] for r in rows}) < 2:
        raise ValueError("trnsfr: the source cohort has no exposure "
                         "variation, so no MSM coefficient is "
                         "identified")
    fit = k.wls(rows, ys, ws)
    return RichResult(payload={
        "estimate": fit["coef"][1], "intercept": fit["coef"][0],
        "coef": fit["coef"], "weights": tot,
        "transport_weights": tw, "msm_weights": msm_w,
        "target": tgt, "cohorts": sorted(set(lab)), "n": n,
        "method": "MSM fitted under IPW weights multiplied by "
                  "cohort-transport weights; Wager (2025) Secs. 2.2 "
                  "and 7.1",
    })


def cheatsheet():
    return ("trnsfr: move an effect between cohorts by reweighting. "
            "p0(x)/p1(x) = [P(S=0)/P(S=1)] (1-pi(x))/pi(x), so a "
            "logistic model for COHORT membership gives the weights -- "
            "no density ratio is modelled. Overlap binds: outside the "
            "source's support nothing is identified, so extreme pi is "
            "refused, not trimmed silently. Routes: ipw / outcome / dr "
            "(default) / balance. Balancing weights match the named "
            "moments EXACTLY even under misspecification; IPW does "
            "not.")


# compact alias per ledger/NAMING.md
transferlearningmsm = transfer_msm
transfer_learning_msm = transfer_msm
