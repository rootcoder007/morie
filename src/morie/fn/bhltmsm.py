# morie.fn -- function file (rootcoder007/morie)
r"""A marginal structural model for cumulative treatment episodes.

**Why the obvious regression is wrong here.** In behavioural health,
whether an adolescent receives another period of treatment depends on
how they are doing -- and how they are doing is itself a consequence of
earlier treatment. Severity at month 3 is therefore a **time-varying
confounder that is affected by prior treatment**. Condition on it and
you block part of the very effect you are estimating; leave it out and
confounding remains. No single regression can do both, which is the
gap marginal structural models exist to fill.

**IPTW re-creates a population in which treatment was assigned at
random.** Weight each person by the inverse probability of the
treatment history they actually received, and fit the outcome model on
that pseudo-population -- *without* the time-varying covariates on the
right-hand side. That the MSM is fitted on cumulative episodes,

.. math:: E[Y] = \alpha_0 + \alpha_{out}\,\mathrm{cum}(out)
          + \alpha_{res}\,\mathrm{cum}(res)
          + \alpha_{bds}\,\mathrm{cum}(bds),

is what makes each coefficient read as "one further period of this
treatment".

**Stabilised weights, and why the mean is a diagnostic.** With
stabilised weights the mean should be close to 1; a mean far from 1,
or a maximum in the hundreds, says a few people are carrying the
estimate and the positivity assumption is strained.
``weight_diagnostics`` reports both rather than truncating quietly,
because truncation trades variance for bias and that trade should be
visible.

**The application.** Griffin et al. followed 2,870 adolescents in 17
community-based programmes over 12 months, with four mutually
exclusive treatment states per 90-day period -- residential,
outpatient, biological drug screening, or none -- propensity weights
from generalised boosted models, and substance-use frequency at 12
months as the outcome. Each additional period reduced use: about 16%
for residential, 11% for screening, 9% for outpatient.

**Ledger note.** This module previously carried the citation "applied
benchmark", which is not a citation, and a body that returned the mean
of the outcome with a standard error. Both are replaced here.

References
----------
Griffin, B. A., Ramchand, R., Almirall, D., Slaughter, M. E.,
Burgette, L. F. & McCaffrey, D. F. (2014) "Estimating the causal
effects of cumulative treatment episodes for adolescents using
marginal structural models and inverse probability of treatment
weighting", *Drug and Alcohol Dependence* 136, 69-78,
doi:10.1016/j.drugalcdep.2013.12.017. The application: 2,870
adolescents from 17 community-based substance abuse treatment
programmes; four mutually exclusive treatment categories per 90-day
period (residential, outpatient, biological drug screening only, or no
treatment); propensity scores for each category estimated by
GENERALIZED BOOSTED MODELS conditional on baseline and time-varying
confounders; a linear MSM in the CUMULATIVE number of episodes of each
type; the Substance Frequency Scale at 12 months as outcome; and
reductions of roughly 16% (residential), 11% (screening) and 9%
(outpatient) per additional period. NOTE: the publisher PDF is
paywalled and the RAND reprint returns 403; the above was verified
against the PubMed Central author manuscript (PMC3969884), not a local
PDF. Crossref renders the last author as "McCaffery"; the correct
spelling is McCaffrey.

Robins, J. M., Hernan, M. A. & Brumback, B. (2000) "Marginal
Structural Models and Causal Inference in Epidemiology",
*Epidemiology* 11(5), 550-560,
doi:10.1097/00001648-200009000-00011. The estimator: that standard
methods are biased when a time-dependent covariate is both a
confounder and affected by previous treatment; marginal structural
models as a class of causal models whose parameters are consistently
estimated by inverse-probability-of-treatment weighted estimators; and
the stabilised weights.

McCaffrey, D. F., Ridgeway, G. & Morral, A. R. (2004) "Propensity
score estimation with boosted regression for evaluating causal effects
in observational studies", *Psychological Methods* 9(4), 403-425,
doi:10.1037/1082-989X.9.4.403. The boosted-regression propensity
weights used, developed on adolescent substance abuse treatment data.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["cumulative_episodes", "treatment_weights",
           "weight_diagnostics", "fit_msm", "confounding_check"]

_EPS = 1e-12
_STATES = ("none", "outpatient", "residential", "screening")


def cumulative_episodes(histories, states=_STATES):
    r"""Count each treatment state over a person's periods.

    The MSM's regressors, so each coefficient reads as the effect of
    ONE further period of that treatment.
    """
    S = list(states)
    idx = {s: i for i, s in enumerate(S)}
    out = []
    for h in histories:
        row = [0.0] * len(S)
        for a in h:
            key = a if isinstance(a, str) else S[int(a)]
            if key not in idx:
                raise ValueError("bhltmsm: unknown treatment state "
                                 "%r; the states are %s"
                                 % (key, ", ".join(S)))
            row[idx[key]] += 1.0
        out.append(row)
    return {"cumulative": out, "states": S,
            "periods": [sum(r) for r in out],
            "note": "mutually exclusive states per period, so the "
                    "counts add to the number of periods"}


def treatment_weights(histories, propensities, stabilise=True,
                      marginal=None, truncate=None):
    r"""The inverse probability of the history actually received.

    ``propensities[i][t]`` is the fitted probability of person
    :math:`i`'s period-:math:`t` treatment given their history and
    covariates. Stabilising multiplies by the same probability from a
    model with no time-varying covariates, which is what keeps the
    weights near 1 without changing what is estimated.
    """
    W, raw = [], []
    for i, h in enumerate(histories):
        P = [float(v) for v in propensities[i]]
        if len(P) != len(h):
            raise ValueError("bhltmsm: person %d has %d periods but "
                             "%d propensities" % (i, len(h), len(P)))
        if any(v <= 0.0 or v > 1.0 for v in P):
            raise ValueError("bhltmsm: a propensity is outside (0,1] "
                             "for person %d -- positivity fails, so "
                             "the weight is undefined" % i)
        den = 1.0
        for v in P:
            den *= v
        num = 1.0
        if stabilise:
            if marginal is None:
                raise ValueError("bhltmsm: stabilised weights need "
                                 "the marginal (no time-varying "
                                 "covariates) probabilities")
            Q = [float(v) for v in marginal[i]]
            if len(Q) != len(P):
                raise ValueError("bhltmsm: the marginal model has a "
                                 "different number of periods")
            for v in Q:
                num *= v
        w = num / den
        raw.append(w)
        W.append(w)
    if truncate is not None:
        lo, hi = k.quantile7(raw, float(truncate)), \
            k.quantile7(raw, 1.0 - float(truncate))
        W = [min(max(v, lo), hi) for v in raw]
    return {"weights": W, "raw": raw, "stabilised": bool(stabilise),
            "truncated": truncate is not None,
            "n_truncated": sum(1 for i in range(len(W))
                               if W[i] != raw[i]),
            "note": "truncation trades variance for BIAS, so what it "
                    "changed is reported"}


def weight_diagnostics(weights):
    r"""Mean near 1, and how much mass sits in the tail.

    A stabilised weight has mean 1 in the population; a sample mean
    far from 1, or a large maximum, means a handful of people are
    carrying the estimate.
    """
    w = [float(v) for v in k.vec(weights)]
    n = len(w)
    if n < 2:
        raise ValueError("bhltmsm: at least 2 weights are needed")
    m = sum(w) / n
    ess = (sum(w) ** 2) / sum(v * v for v in w)
    return {"mean": m, "max": max(w), "min": min(w),
            "effective_n": ess, "n": n,
            "efficiency": ess / n,
            "mean_near_one": abs(m - 1.0) < 0.1,
            "note": "effective sample size collapses when a few "
                    "weights dominate, which is what a positivity "
                    "violation looks like"}


def confounding_check(covariate_history, treatment_history,
                      outcome=None):
    r"""Is the covariate a confounder AFFECTED by prior treatment?

    The condition that rules out ordinary regression: if the answer
    is yes, adjusting blocks part of the effect and not adjusting
    leaves confounding.
    """
    L = [[float(v) for v in r] for r in k.mat(covariate_history)]
    A = [[float(v) for v in r] for r in k.mat(treatment_history)]
    if len(L) != len(A):
        raise ValueError("bhltmsm: %d covariate histories but %d "
                         "treatment histories" % (len(L), len(A)))
    T = len(L[0])
    if T < 2:
        raise ValueError("bhltmsm: at least 2 periods are needed to "
                         "ask whether treatment affects the "
                         "covariate")
    prior_to_l = k.corr([A[i][0] for i in range(len(A))],
                        [L[i][1] for i in range(len(L))])
    l_to_next_a = k.corr([L[i][0] for i in range(len(L))],
                         [A[i][1] for i in range(len(A))])
    both = abs(prior_to_l) > 0.1 and abs(l_to_next_a) > 0.1
    out = {"treatment_affects_covariate": prior_to_l,
           "covariate_predicts_treatment": l_to_next_a,
           "is_treatment_confounder_feedback": both,
           "note": "both arrows present means neither adjusting nor "
                   "not adjusting is valid -- hence IPTW"}
    if outcome is not None:
        y = [float(v) for v in k.vec(outcome)]
        out["covariate_predicts_outcome"] = k.corr(
            [L[i][1] for i in range(len(L))], y)
    return out


def fit_msm(outcome, cumulative, weights=None, states=_STATES):
    r"""Weighted least squares of the outcome on cumulative episodes.

    The time-varying covariates are deliberately ABSENT from the right
    hand side -- they were used to build the weights, and putting them
    here as well would block the mediated part of the effect.
    """
    y = [float(v) for v in k.vec(outcome)]
    X = [[float(v) for v in r] for r in k.mat(cumulative)]
    n = len(y)
    if len(X) != n:
        raise ValueError("bhltmsm: %d outcomes but %d covariate rows"
                         % (n, len(X)))
    w = [1.0] * n if weights is None else [float(v)
                                           for v in k.vec(weights)]
    if len(w) != n:
        raise ValueError("bhltmsm: %d weights for %d people"
                         % (len(w), n))
    if any(v < 0.0 for v in w):
        raise ValueError("bhltmsm: a weight is negative")
    co = k.wls(X, y, w, 1e-10)["coef"]
    fit = [co[0] + sum(X[i][a] * co[1 + a] for a in range(len(X[0])))
           for i in range(n)]
    res = [y[i] - fit[i] for i in range(n)]
    dof = max(n - len(X[0]) - 1, 1)
    s2 = sum(w[i] * res[i] ** 2 for i in range(n)) / dof
    ses = []
    for a in range(len(X[0])):
        xm = sum(w[i] * X[i][a] for i in range(n)) / sum(w)
        sxx = sum(w[i] * (X[i][a] - xm) ** 2 for i in range(n))
        ses.append(math.sqrt(s2 / sxx) if sxx > _EPS
                   else float("inf"))
    names = list(states)[:len(X[0])]
    return RichResult(payload={
        "estimate": co[1], "intercept": co[0],
        "coefficients": dict(zip(names, co[1:])),
        "se": dict(zip(names, ses)),
        "per_episode": dict(zip(names, co[1:])),
        "weighted": weights is not None,
        "effective_n": (sum(w) ** 2) / sum(v * v for v in w),
        "method": "marginal structural model by IPTW; Robins, Hernan "
                  "& Brumback (2000), applied as in Griffin et al. "
                  "(2014)",
        "note": "the time-varying covariates are NOT regressors here; "
                "they built the weights",
    })


def cheatsheet():
    return ("bhltmsm: whether someone gets another period of treatment "
            "depends on how they are doing, and how they are doing is "
            "a CONSEQUENCE of earlier treatment -- a time-varying "
            "confounder AFFECTED BY prior treatment. Adjust for it and "
            "you block part of the effect; omit it and confounding "
            "stays. So weight each person by the inverse probability "
            "of the treatment history they received and fit the "
            "outcome on CUMULATIVE episodes with the covariates "
            "ABSENT. STABILISE the weights (mean near 1) and read the "
            "effective sample size: a collapsing ESS is what a "
            "positivity violation looks like. Truncation trades "
            "variance for bias -- report it.")


# compact alias per ledger/NAMING.md
behavioral_health_msm = fit_msm
