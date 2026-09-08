# morie.fn -- function file (rootcoder007/morie)
r"""CV-TMLE for the mean outcome under an optimal dynamic treatment rule.

Two time points. We see

.. math:: O = (L(0), A(0), L(1), A(1), Y) \sim P_0,

and a dynamic rule assigns each treatment from the history available at
that point: :math:`d_{A(0)}` may look at :math:`V(0)`, a function of
:math:`L(0)`, and :math:`d_{A(1)}` at :math:`(A(0), V(1))`. Restricting
the rule to a user-chosen summary :math:`V` is deliberate -- the
**V-optimal** rule is easier to estimate than the fully optimal one and
admits inference for its own mean outcome.

**The rule comes from backward induction, not from a search.** Theorem
22.1 writes the V-optimal rule in closed form through two blip
functions,

.. math::
   \bar Q_{b,2}(a(0), v(1)) &= E[Y_{a(0),A(1)=1} \mid V_{a(0)}(1)=v(1)]
                              - E[Y_{a(0),A(1)=0} \mid V_{a(0)}(1)=v(1)],\\
   d_{A(1)}(a(0), v(1)) &= I(\bar Q_{b,2}(a(0), v(1)) > 0),\\
   \bar Q_{b,1}(v(0)) &= E[Y_{1,d_{A(1)}} \mid V(0)=v(0)]
                        - E[Y_{0,d_{A(1)}} \mid V(0)=v(0)],\\
   d_{A(0)}(v(0)) &= I(\bar Q_{b,1}(v(0)) > 0).

The second-stage rule is solved first and then *carried into* the
first-stage contrast. Take the shortcut of contrasting :math:`A(0)`
under a fixed :math:`A(1)` and the first-stage rule stops accounting for
what it will be able to do at time 1 -- which is the entire content of a
dynamic regime.

**Why the estimator is cross-validated.** The rule is itself estimated,
so plugging it into a target parameter and then evaluating that
parameter on the same data is the mistake the CV-TMLE of Sec. 22.6
exists to avoid: the rule is fitted on the training split, and the
targeting and the mean are evaluated on the validation split. The two
fluctuations

.. math:: H_2(g)(O) = \frac{I(\bar A(1) = d(A(0), V))}
                           {\prod_{l=0}^{1} g_{A(l)}(O)},
          \qquad
          H_1(g)(O) = \frac{I(A(0) = d_{A(0)}(V(0)))}{g_{A(0)}(O)}

are fitted once across the folds -- each :math:`\epsilon` is a single
scalar, which is what keeps the empirical-process conditions off the
initial fits -- and the result solves the cross-validated efficient
influence curve equation exactly.

**Exceptional laws.** Eq. (22.5) requires both blip functions to be
non-zero almost surely. Where a blip is flat at zero the argmax is not
unique, the rule is unstable and the influence-curve-based interval does
not hold. ``exceptional_law_share`` reports how much mass sits near zero
rather than leaving it to be discovered as a coverage failure.

References
----------
Luedtke, A. R. & van der Laan, M. J. (2018) "Optimal Dynamic Treatment
Rules", Ch. 22 in van der Laan, M. J. & Rose, S. (eds.) *Targeted
Learning in Data Science: Causal Inference for Complex Longitudinal
Studies*, Springer Series in Statistics, pp. 399-419,
doi:10.1007/978-3-319-65304-4_22. Theorem 22.1 (the blip
representation), eq. (22.4) (the efficient influence curve), Sec. 22.6
(the CV-TMLE implemented here).

van der Laan, M. J. & Luedtke, A. R. (2015) "Targeted Learning of the
Mean Outcome Under an Optimal Dynamic Treatment Rule", *Journal of
Causal Inference* 3(1), 61-95, doi:10.1515/jci-2013-0022.

Robins, J. M. (2004) "Optimal Structural Nested Models for Optimal
Sequential Decisions", in Lin, D. Y. & Heagerty, P. J. (eds.)
*Proceedings of the Second Seattle Symposium in Biostatistics*, Lecture
Notes in Statistics 179, Springer, pp. 189-326,
doi:10.1007/978-1-4419-9076-1_11. Source of the term "blip function"
and of the exceptional-law condition.

Zheng, W. & van der Laan, M. J. (2011) "Cross-Validated Targeted
Minimum-Loss-Based Estimation", in van der Laan, M. J. & Rose, S.
*Targeted Learning*, Springer Series in Statistics, pp. 459-474,
doi:10.1007/978-1-4419-9782-1_27. The CV-TMLE template Sec. 22.6
modifies.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmle_dynamic_regime", "sequential_blips", "optimal_rule",
           "intervention_mechanism", "rule_value_seq"]

_METHODS = ("cv-tmle", "tmle", "ipw", "gcomp")
_EPS = 1e-9


def _logit(p):
    q = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(q / (1.0 - q))


def _expit(z):
    return k.sigmoid(z)


def _blocks(covariate_history, n):
    """Split the covariate history into the L(0) and L(1) blocks."""
    if covariate_history is None:
        raise ValueError("tmldyn: covariate_history is required")
    ch = list(covariate_history)
    if len(ch) != 2:
        raise ValueError("tmldyn: covariate_history must be two blocks "
                         "[L0, L1], got %d" % len(ch))
    L0, L1 = k.mat(ch[0]), k.mat(ch[1])
    if len(L0) != n or len(L1) != n:
        raise ValueError("tmldyn: covariate blocks have %d and %d rows "
                         "but there are %d outcomes"
                         % (len(L0), len(L1), n))
    return L0, L1


def _project(values, basis, n, ridge):
    """E[values | V] by least squares on the summary V."""
    if basis is None:
        return list(values)
    Z = k.design(k.mat(basis), n)
    return list(k.matvec(Z, k.lstsq(Z, k.vec(values), ridge)))


def intervention_mechanism(L0, A0, L1, A1, trim=0.01, known=None,
                           penalty=0.0):
    r"""g_{A(0)}(O) and g_{A(1)}(O), the intervention mechanism.

    ``known`` supplies the true probabilities of receiving treatment --
    the SMART case, where the design fixes them and nothing needs to be
    estimated. Pass ``(p0, p1)`` with ``p_j[i] = P(A(j)=1 | history)``.
    """
    n = len(A0)
    if known is not None:
        p0, p1 = [float(v) for v in known[0]], [float(v) for v in known[1]]
        if len(p0) != n or len(p1) != n:
            raise ValueError("tmldyn: known g has the wrong length")
    else:
        # logit_irls returns the coefficients, so the probabilities are
        # formed here rather than read off a second return value.
        X0 = k.design(L0, n)
        p0 = [_expit(v) for v in
              k.matvec(X0, k.logit_irls(X0, A0, penalty=penalty))]
        X1 = k.design([[A0[i]] + list(L0[i]) + list(L1[i])
                       for i in range(n)], n)
        p1 = [_expit(v) for v in
              k.matvec(X1, k.logit_irls(X1, A1, penalty=penalty))]
    t = float(trim)
    if not 0.0 <= t < 0.5:
        raise ValueError("tmldyn: trim must be in [0, 0.5), got %r"
                         % (trim,))
    lo, hi = max(t, _EPS), 1.0 - max(t, _EPS)
    p0 = [min(max(v, lo), hi) for v in p0]
    p1 = [min(max(v, lo), hi) for v in p1]
    # the probability of the treatment actually received
    g0 = [p0[i] if A0[i] == 1.0 else 1.0 - p0[i] for i in range(n)]
    g1 = [p1[i] if A1[i] == 1.0 else 1.0 - p1[i] for i in range(n)]
    return g0, g1, {"p0": p0, "p1": p1,
                    "min_g0": min(g0), "min_g1": min(g1),
                    "max_weight": max(1.0 / (g0[i] * g1[i])
                                      for i in range(n)),
                    "known": known is not None}


def _fit_q2(y, L0, A0, L1, A1, idx, ridge):
    """E[Y | Abar(1), Lbar(1)], fitted on the rows in idx.

    Treatment interacts with both covariate blocks, and the two
    treatments interact with each other: without those terms the blips
    are constants and the optimal rule is static by construction.
    """
    def row(a0, a1, i):
        return ([1.0, a0, a1, a0 * a1] + list(L0[i]) + list(L1[i])
                + [a1 * v for v in L1[i]] + [a1 * v for v in L0[i]]
                + [a0 * v for v in L0[i]])

    X = [row(A0[i], A1[i], i) for i in idx]
    b = k.lstsq(X, [y[i] for i in idx], ridge)

    def q2(a0, a1, i):
        r = row(a0, a1, i)
        return sum(b[t] * r[t] for t in range(len(b)))

    return q2, b


def _fit_q1(pseudo, L0, A0, idx, ridge):
    """E[ Q2(A(0), d_{A(1)}, Lbar(1)) | A(0), L(0) ], on the rows in idx."""
    def row(a0, i):
        return [1.0, a0] + list(L0[i]) + [a0 * v for v in L0[i]]

    X = [row(A0[i], i) for i in idx]
    b = k.lstsq(X, [pseudo[i] for i in idx], ridge)

    def q1(a0, i):
        r = row(a0, i)
        return sum(b[t] * r[t] for t in range(len(b)))

    return q1, b


def sequential_blips(y, L0, A0, L1, A1, V0=None, V1=None, ridge=1e-8,
                     idx=None, eval_idx=None):
    r"""Theorem 22.1: the two blip functions and the V-optimal rule.

    Fitted on ``idx`` (all rows by default) and evaluated on
    ``eval_idx``. Splitting the two is what makes the cross-validated
    estimator possible: the rule must not be read off the same rows it
    is then scored on.

    Returns
    -------
    dict
        ``blip2`` is indexed ``[a0][i]`` because the second-stage blip
        is allowed to depend on the first treatment; ``blip1`` and
        ``d0`` are per-subject; ``d1[a0][i]`` is the second-stage
        assignment under each possible first-stage treatment.
    """
    n = len(y)
    idx = list(range(n)) if idx is None else list(idx)
    eval_idx = list(range(n)) if eval_idx is None else list(eval_idx)
    q2, b2 = _fit_q2(y, L0, A0, L1, A1, idx, ridge)

    # ---- stage 2: the blip under each possible first treatment
    raw2 = [[q2(a0, 1.0, i) - q2(a0, 0.0, i) for i in range(n)]
            for a0 in (0.0, 1.0)]
    basis1 = V1 if V1 is not None else L1
    blip2 = [_project(raw2[a], basis1, n, ridge) for a in (0, 1)]
    d1 = [[1.0 if blip2[a][i] > 0.0 else 0.0 for i in range(n)]
          for a in (0, 1)]

    # ---- carry the stage-2 rule into the stage-1 contrast
    pseudo = [q2(A0[i], d1[int(A0[i])][i], i) for i in range(n)]
    q1, b1 = _fit_q1(pseudo, L0, A0, idx, ridge)
    raw1 = [q1(1.0, i) - q1(0.0, i) for i in range(n)]
    basis0 = V0 if V0 is not None else L0
    blip1 = _project(raw1, basis0, n, ridge)
    d0 = [1.0 if v > 0.0 else 0.0 for v in blip1]

    return {"blip1": blip1, "blip2": blip2, "d0": d0, "d1": d1,
            "q2": q2, "q1": q1, "coef_q2": b2, "coef_q1": b1,
            "pseudo": pseudo, "eval_idx": eval_idx}


def optimal_rule(y, L0, A0, L1, A1, V0=None, V1=None, ridge=1e-8):
    """The estimated V-optimal rule, as ``(d0, d1)``."""
    r = sequential_blips(y, L0, A0, L1, A1, V0=V0, V1=V1, ridge=ridge)
    return r["d0"], r["d1"]


def exceptional_law_share(blips, tol=0.01):
    r"""Share of subjects whose blip sits within ``tol`` of zero.

    Eq. (22.5) rules these laws out. A large share means the argmax is
    not well separated, so the rule is unstable and the interval below
    should not be trusted -- worth reporting rather than discovering.
    """
    v = [abs(float(b)) for b in blips]
    return sum(1 for b in v if b <= tol) / float(len(v)) if v else 0.0


def _fluctuate(outcome, offset_logit, H, rows, iters=100, tol=1e-12):
    """One-dimensional logistic fluctuation, eq. (22.12).

    Solves sum_i H_i (Y_i - expit(offset_i + eps H_i)) = 0 by Newton.
    This is a univariate logistic regression of ``outcome`` on ``H``
    with ``offset_logit`` as offset, which is exactly how Sec. 22.6
    words it.
    """
    if not rows or all(abs(H[i]) < 1e-14 for i in rows):
        return 0.0
    eps = 0.0
    for _ in range(iters):
        num = den = 0.0
        for i in rows:
            p = _expit(offset_logit[i] + eps * H[i])
            num += H[i] * (outcome[i] - p)
            den += H[i] * H[i] * p * (1.0 - p)
        if den < 1e-14:
            break
        step = num / den
        eps += step
        if abs(step) < tol:
            break
    return eps


def _folds(n, n_folds, seed=0):
    """Deterministic contiguous-stride folds."""
    J = max(2, min(int(n_folds), n))
    return [[i for i in range(n) if i % J == j] for j in range(J)]


def rule_value_seq(y, L0, A0, L1, A1, d0, d1, g0, g1, ridge=1e-8):
    """Plug-in value of a *given* rule by sequential regression.

    No targeting: this is the g-computation arm, kept so the effect of
    the fluctuation is visible rather than asserted.
    """
    n = len(y)
    q2, _ = _fit_q2(y, L0, A0, L1, A1, list(range(n)), ridge)
    pseudo = [q2(A0[i], d1[int(A0[i])][i], i) for i in range(n)]
    q1, _ = _fit_q1(pseudo, L0, A0, list(range(n)), ridge)
    return sum(q1(d0[i], i) for i in range(n)) / n


def _coerce_regime(regime, n):
    """A supplied rule, as (d0, d1[a0][i])."""
    if regime is None or (isinstance(regime, str)
                          and regime.lower() in ("optimal", "v-optimal")):
        return None
    if isinstance(regime, str):
        raise ValueError("tmldyn: regime must be 'optimal' or an array, "
                         "got %r" % (regime,))
    r = list(regime)
    if len(r) == 2 and hasattr(r[0], "__len__") and len(r[0]) == n:
        d0 = [float(v) for v in r[0]]
        second = r[1]
        # (d0, d1) with d1 either n-long (no dependence on a(0)) or the
        # two branches under a(0) = 0 and a(0) = 1
        if len(second) == 2 and hasattr(second[0], "__len__") \
                and len(second[0]) == n:
            d1 = [[float(v) for v in second[a]] for a in (0, 1)]
        elif len(second) == n:
            col = [float(v) for v in second]
            d1 = [list(col), list(col)]
        else:
            raise ValueError("tmldyn: regime's second component has "
                             "length %d, expected %d or 2"
                             % (len(second), n))
        return d0, d1
    if len(r) == n:                       # n-by-2 table of assignments
        d0 = [float(row[0]) for row in r]
        col = [float(row[1]) for row in r]
        return d0, [list(col), list(col)]
    raise ValueError("tmldyn: cannot read regime of length %d for n = %d"
                     % (len(r), n))


def tmle_dynamic_regime(y, treatment_history, covariate_history,
                        regime="optimal", method="cv-tmle", n_folds=10,
                        V0=None, V1=None, trim=0.01, known_g=None,
                        ridge=1e-8, level=0.95):
    r"""Mean outcome under the (V-)optimal dynamic treatment rule.

    Parameters
    ----------
    y : array-like
        Final outcome, length n. Rescaled to [0, 1] internally, as
        Sec. 22.6 assumes without loss of generality; everything
        reported is on the original scale.
    treatment_history : array-like
        n-by-2 of binary ``A(0), A(1)``.
    covariate_history : sequence of two array-likes
        ``[L(0), L(1)]``, each n rows.
    regime : {"optimal"} or array-like
        ``"optimal"`` estimates the V-optimal rule by Theorem 22.1.
        Otherwise a supplied rule: ``(d0, d1)`` where ``d1`` is either
        n-long or the two branches under ``A(0) = 0`` and ``1``, or an
        n-by-2 table of assignments.
    method : {"cv-tmle", "tmle", "ipw", "gcomp"}
        ``cv-tmle`` is Sec. 22.6 and the default: the rule and the
        nuisance fits come from the training split and the targeting
        from the validation split. ``tmle`` does the same fluctuation
        without the split, which is faster and biased when the rule is
        estimated. ``ipw`` is the Horvitz-Thompson mean under the rule
        and ``gcomp`` the untargeted sequential regression; both are
        kept so the fluctuation's effect is measurable.
    n_folds : int
        Number of cross-validation splits, ``cv-tmle`` only.
    V0, V1 : array-like, optional
        The summaries the two rules may depend on. Default to the full
        covariate blocks, giving the fully optimal rule within the
        model.
    trim : float
        Lower bound on the intervention mechanism, for positivity.
    known_g : (p0, p1), optional
        Known treatment probabilities, as in a SMART.

    Returns
    -------
    RichResult
        ``estimate`` is the mean outcome under the rule, with ``se`` and
        ``ci`` from the efficient influence curve, the fitted rule, the
        blips, and the comparison arms.

    Examples
    --------
    Estimate the rule and its value together::

        r = tmle_dynamic_regime(y, A, [L0, L1])
        r["estimate"], r["ci"], r["exceptional_share_1"]

    Score a rule that was fixed in advance::

        r = tmle_dynamic_regime(y, A, [L0, L1], regime=(d0, d1))
    """
    if method not in _METHODS:
        raise ValueError("tmldyn: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    yv = k.vec(y)
    n = len(yv)
    if n < 4:
        raise ValueError("tmldyn: need at least 4 observations, got %d"
                         % n)
    Am = k.mat(treatment_history)
    if len(Am) != n or len(Am[0]) != 2:
        raise ValueError("tmldyn: treatment_history must be n-by-2, "
                         "got %d-by-%d" % (len(Am), len(Am[0])))
    A0 = [float(r[0]) for r in Am]
    A1 = [float(r[1]) for r in Am]
    if any(v not in (0.0, 1.0) for v in A0 + A1):
        raise ValueError("tmldyn: treatments must be binary 0/1")
    L0, L1 = _blocks(covariate_history, n)

    ymin, ymax = min(yv), max(yv)
    rng = ymax - ymin
    if rng <= 0.0:
        raise ValueError("tmldyn: the outcome is constant")
    ys = [(v - ymin) / rng for v in yv]

    g0, g1, ginfo = intervention_mechanism(L0, A0, L1, A1, trim=trim,
                                           known=known_g)
    supplied = _coerce_regime(regime, n)

    # ---------------------------------------------------- the rule
    if supplied is not None:
        d0, d1 = supplied
        full = sequential_blips(ys, L0, A0, L1, A1, V0=V0, V1=V1,
                                ridge=ridge)
        blip1, blip2 = full["blip1"], full["blip2"]
        splits = [(list(range(n)), list(range(n)))]
        rules = [(d0, d1)]
    elif method == "cv-tmle":
        splits, rules = [], []
        d0 = [0.0] * n
        d1 = [[0.0] * n, [0.0] * n]
        blip1 = [0.0] * n
        blip2 = [[0.0] * n, [0.0] * n]
        for val in _folds(n, n_folds):
            train = [i for i in range(n) if i not in set(val)]
            fit = sequential_blips(ys, L0, A0, L1, A1, V0=V0, V1=V1,
                                   ridge=ridge, idx=train)
            splits.append((train, val))
            rules.append((fit["d0"], fit["d1"]))
            for i in val:                       # the rule this row gets
                d0[i] = fit["d0"][i]
                blip1[i] = fit["blip1"][i]
                for a in (0, 1):
                    d1[a][i] = fit["d1"][a][i]
                    blip2[a][i] = fit["blip2"][a][i]
    else:
        fit = sequential_blips(ys, L0, A0, L1, A1, V0=V0, V1=V1,
                               ridge=ridge)
        d0, d1 = fit["d0"], fit["d1"]
        blip1, blip2 = fit["blip1"], fit["blip2"]
        splits = [(list(range(n)), list(range(n)))]
        rules = [(d0, d1)]

    # ------------------------------------------- clever covariates
    # H2 needs the WHOLE history to follow the rule, H1 only A(0).
    follow0 = [1.0 if A0[i] == d0[i] else 0.0 for i in range(n)]
    follow1 = [1.0 if A1[i] == d1[int(A0[i])][i] else 0.0
               for i in range(n)]
    H1 = [follow0[i] / g0[i] for i in range(n)]
    H2 = [follow0[i] * follow1[i] / (g0[i] * g1[i]) for i in range(n)]

    if method == "ipw":
        psi_s = sum(H2[i] * ys[i] for i in range(n)) / n
        eic = [H2[i] * ys[i] - psi_s for i in range(n)]
        q2d = [ys[i] for i in range(n)]
        q1d = [psi_s] * n
    else:
        # initial fits, evaluated at the rule; under cv-tmle the fit for
        # a row comes from the split that did not contain it
        q2d = [0.0] * n
        q1d = [0.0] * n
        for (train, val), (rd0, rd1) in zip(splits, rules):
            q2, _ = _fit_q2(ys, L0, A0, L1, A1, train, ridge)
            pseudo = [q2(A0[i], rd1[int(A0[i])][i], i) for i in range(n)]
            q1, _ = _fit_q1(pseudo, L0, A0, train, ridge)
            for i in val:
                q2d[i] = min(max(q2(rd0[i], rd1[int(rd0[i])][i], i),
                                 _EPS), 1.0 - _EPS)
                q1d[i] = min(max(q1(rd0[i], i), _EPS), 1.0 - _EPS)

        if method == "gcomp":
            psi_s = sum(q1d) / n
            eic = [q1d[i] - psi_s for i in range(n)]
            eps2 = eps1 = 0.0
        else:
            # Sec. 22.6: one scalar epsilon each, pooled over the folds
            off2 = [_logit(v) for v in q2d]
            eps2 = _fluctuate(ys, off2, H2, list(range(n)))
            q2d = [_expit(off2[i] + eps2 * H2[i]) for i in range(n)]
            off1 = [_logit(v) for v in q1d]
            eps1 = _fluctuate(q2d, off1, H1, list(range(n)))
            q1d = [_expit(off1[i] + eps1 * H1[i]) for i in range(n)]
            psi_s = sum(q1d) / n
            eic = [(q1d[i] - psi_s)
                   + H1[i] * (q2d[i] - q1d[i])
                   + H2[i] * (ys[i] - q2d[i]) for i in range(n)]

    psi = ymin + rng * psi_s
    se = k.sd(eic) * rng / math.sqrt(n) if n > 1 else float("nan")
    z = k.qnorm(0.5 + 0.5 * float(level))

    # ------------------------------------------------- comparators
    # Every static regime, so "the dynamic rule is better" is a claim
    # that can be checked rather than assumed.
    static = {}
    for a0 in (0.0, 1.0):
        for a1 in (0.0, 1.0):
            v = rule_value_seq(ys, L0, A0, L1, A1, [a0] * n,
                               [[a1] * n, [a1] * n], g0, g1, ridge)
            static["static_%d%d" % (int(a0), int(a1))] = ymin + rng * v

    return RichResult(payload={
        "estimate": psi, "se": se, "n": n,
        "ci": (psi - z * se, psi + z * se),
        "level": float(level),
        "d0": d0, "d1": d1, "blip1": blip1, "blip2": blip2,
        "treated_first": sum(d0) / n,
        "treated_second": sum(d1[int(A0[i])][i] for i in range(n)) / n,
        "eic_mean": sum(eic) / n,
        "epsilon": (locals().get("eps1", 0.0),
                    locals().get("eps2", 0.0)),
        "max_weight": ginfo["max_weight"],
        "min_g0": ginfo["min_g0"], "min_g1": ginfo["min_g1"],
        "known_g": ginfo["known"],
        "exceptional_share_1": exceptional_law_share(blip1),
        "exceptional_share_2": max(exceptional_law_share(blip2[0]),
                                   exceptional_law_share(blip2[1])),
        "value_gcomp": ymin + rng * (sum(q1d) / n),
        "best_static": max(static.values()),
        "n_folds": len(splits), "method": method,
        "rule_source": "supplied" if supplied is not None else "estimated",
        "algorithm": "CV-TMLE for the mean outcome under the V-optimal "
                     "dynamic rule, Luedtke & van der Laan (2018) "
                     "Thm 22.1 and Sec. 22.6",
        **static,
    })


def cheatsheet():
    return ("tmldyn: two time points. Backward induction (Thm 22.1): "
            "Qb2(a0,v1)=E[Y_{a0,1}-Y_{a0,0}|V(1)], d1=I(Qb2>0); carry "
            "d1 into Qb1(v0)=E[Y_{1,d1}-Y_{0,d1}|V(0)], d0=I(Qb1>0). "
            "Then CV-TMLE (Sec 22.6): H2=I(Abar=d)/(g0 g1), "
            "H1=I(A0=d0)/g0, one scalar epsilon each, rule from the "
            "training split and the mean from the validation split.")


# compact alias per ledger/NAMING.md
tmledynamicregime = tmle_dynamic_regime
