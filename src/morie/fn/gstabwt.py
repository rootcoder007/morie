# morie.fn -- function file (rootcoder007/morie)
"""Stabilized inverse-probability-of-treatment weights.

Robins, J.M., Hernan, M.A. & Brumback, B. (2000).  Marginal
structural models and causal inference in epidemiology.
Epidemiology 11:550-560.

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

from ._richresult import RichResult, with_describe_pointer

__all__ = ["stabilized_weights"]


def _prod_rows(P):
    """Product over the time index for each subject."""
    out = []
    for row in P:
        v = 1.0
        for p in row:
            v *= float(p)
        out.append(v)
    return out


def stabilized_weights(treatment=None, history=None, numerator_model=None,
                       denominator_model=None):
    """Stabilized IPT weights over a treatment history,

        sw_i = prod_t f(A_it | A_i,t-1) / f(A_it | H_it).

    The unstabilized weight is the denominator product alone.  Putting
    the marginal treatment probability in the numerator leaves the
    weights centred near one instead of letting a single small
    conditional probability blow the weight up, which is the whole
    point of stabilizing (handbook ch.18 p.364).

    ``numerator_model`` and ``denominator_model`` are the fitted
    probabilities f(A_t | A_t-1) and f(A_t | H_t), one row per
    subject and one column per time point.  They are supplied rather
    than fitted here so the function is deterministic and does not
    hide a nuisance-model choice.  ``treatment`` and ``history`` are
    accepted for signature compatibility and are only used to check
    shapes.

    Diagnostic.  Correctly specified stabilized weights have mean
    close to 1; ``mean_weight`` is returned precisely so that check
    can be made, and a mean far from 1 signals a misspecified
    numerator or a positivity violation.

    Returns
    -------
    RichResult with keys estimate (the mean stabilized weight),
    weights, unstabilized, mean_weight, max_weight, n, n_times,
    method.
    """
    if denominator_model is None:
        raise ValueError("denominator_model (f(A_t | H_t)) is required")
    D = [[float(v) for v in row] for row in denominator_model]
    n = len(D)
    if n == 0:
        raise ValueError("need at least one subject")
    nt = len(D[0])
    if any(len(r) != nt for r in D):
        raise ValueError("denominator_model must be rectangular")
    if any(p <= 0.0 for r in D for p in r):
        raise ValueError("denominator probabilities must be positive")
    if numerator_model is None:
        N = [[1.0] * nt for _ in range(n)]
    else:
        N = [[float(v) for v in row] for row in numerator_model]
        if len(N) != n or any(len(r) != nt for r in N):
            raise ValueError("numerator_model must match denominator_model")
    if treatment is not None and len(list(treatment)) != n:
        raise ValueError("treatment must have one row per subject")
    den = _prod_rows(D)
    num = _prod_rows(N)
    w = [num[i] / den[i] for i in range(n)]
    uw = [1.0 / den[i] for i in range(n)]
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(w) / n), "weights": w,
        "unstabilized": uw, "mean_weight": float(sum(w) / n),
        "max_weight": float(max(w)), "n": n, "n_times": nt,
        "method": "stabilized IPT weights (Robins, Hernan & Brumback 2000)",
    }), "gstabwt")


def cheatsheet():
    return "gstabwt: Stabilized inverse-probability-of-treatment weights"


# compact alias per ledger/NAMING.md
stabwt = stabilized_weights
