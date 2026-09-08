# morie.fn -- function file (rootcoder007/morie)
"""Augmented IPW estimator and its efficient influence function.

Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994).  Estimation of
regression coefficients when some regressors are not always observed.
JASA 89:846-866 -- the augmented estimating equation.

The estimator implemented is eq. (23.4) p.512 of the Handbook of
Matching and Weighting Adjustments for Causal Inference, read from
the corpus PDF and quoted here in full:

    Delta = (1/n) sum_i { W_i (Y_i - m1_i) / p1_i + m1_i
                        - (1 - W_i)(Y_i - m0_i) / p0_i - m0_i }

with p_wi = P(W_i = w | X = X_i) the propensity and
m_wi = E(Y | W = w, X = X_i) the outcome model.

Sources consulted, not recalled:
  Zubizarreta, Stuart, Small & Rosenbaum (eds), Handbook of Matching
  and Weighting Adjustments for Causal Inference, Chapman & Hall/CRC.
  Read from the corpus PDF: the inverse-probability weight and its
  stabilized form, ch.18 p.364; the augmented IPW estimator, eq.
  (23.4) p.512.
  Gruber, S. & van der Laan, M.J. (2012).  tmle: An R Package for
  Targeted Maximum Likelihood Estimation.  J Stat Softw 51(13).
  Fetched and read: the clever covariates (2)-(3) and the logistic
  fluctuation of the targeting step, pp.5-6.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["aipw_efficient_influence"]


def aipw_efficient_influence(y, D, X=None, ml_outcome=None,
                             ml_propensity=None):
    """Doubly robust ATE with its influence-function standard error.

    ``ml_outcome`` is the pair (m1, m0) of fitted outcome predictions
    under treatment and control for every subject; ``ml_propensity``
    is the fitted P(D = 1 | X).  They are supplied, not fitted here,
    so the estimator is deterministic and the nuisance choice stays
    with the caller -- which is also what makes cross-fitting
    possible without this function knowing about it.

    Double robustness: the estimator is consistent if either the
    outcome model or the propensity model is right, not necessarily
    both.  Two identities follow directly from the formula and are
    worth knowing as checks -- with m1 = m0 = 0 it collapses to the
    Horvitz-Thompson IPW difference, and with a constant propensity
    of 1/2 and an outcome model fitted to the arm means it collapses
    to the plain difference in means.

    Returns
    -------
    RichResult with keys estimate (the ATE), se, ci_lower, ci_upper,
    influence, ipw, plugin, n, method.
    """
    ys = [float(v) for v in y]
    dd = [float(v) for v in D]
    n = len(ys)
    if n == 0:
        raise ValueError("need at least one observation")
    if len(dd) != n:
        raise ValueError("y and D must have the same length")
    if ml_outcome is None or ml_propensity is None:
        raise ValueError("ml_outcome (m1, m0) and ml_propensity are required")
    m1, m0 = ml_outcome
    m1 = [float(v) for v in m1]
    m0 = [float(v) for v in m0]
    e = [float(v) for v in ml_propensity]
    if not (len(m1) == len(m0) == len(e) == n):
        raise ValueError("nuisance predictions must have length n")
    if any(not 0.0 < p < 1.0 for p in e):
        raise ValueError("propensities must lie strictly in (0, 1)")
    inf = []
    for i in range(n):
        inf.append(dd[i] * (ys[i] - m1[i]) / e[i] + m1[i]
                   - (1.0 - dd[i]) * (ys[i] - m0[i]) / (1.0 - e[i]) - m0[i])
    est = sum(inf) / n
    var = sum((v - est) ** 2 for v in inf) / (n * n) if n > 1 else float("nan")
    se = math.sqrt(var) if var == var else float("nan")
    z = 1.959963984540054
    ipw = sum(dd[i] * ys[i] / e[i] - (1.0 - dd[i]) * ys[i] / (1.0 - e[i])
              for i in range(n)) / n
    return with_describe_pointer(RichResult(payload={
        "estimate": float(est), "se": float(se),
        "ci_lower": float(est - z * se), "ci_upper": float(est + z * se),
        "influence": inf, "ipw": float(ipw),
        "plugin": float(sum(m1[i] - m0[i] for i in range(n)) / n),
        "n": n,
        "method": "augmented IPW ATE (Robins, Rotnitzky & Zhao 1994)",
    }), "eaiprl")


def cheatsheet():
    return "eaiprl: AIPW efficient influence function ATE"


# compact alias per ledger/NAMING.md
aipwate = aipw_efficient_influence
