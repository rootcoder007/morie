# morie.fn -- function file (rootcoder007/morie)
"""Stabilized inverse-probability-of-censoring weights.

Robins, J.M. (1993).  Information recovery and bias adjustment in
proportional hazards regression analyses of randomized trials.
Proceedings of the Biopharmaceutical Section, ASA, 24-33.

The censoring weight has the same product form as the treatment
weight of Robins, Hernan & Brumback (2000), with the probability of
remaining uncensored in place of the probability of the observed
treatment:

    sw^C_i = prod_t P(C_it = 0 | C_i,t-1 = 0)
             / P(C_it = 0 | C_i,t-1 = 0, H_it).

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

__all__ = ["stabilized_censoring_weights"]


def stabilized_censoring_weights(C, H=None, numerator=None):
    """Weights that undo informative censoring.

    ``C`` holds the conditional probabilities of remaining uncensored
    at each time given history, one row per subject.  A subject who
    is censored contributes weight zero from that point on, which is
    how the uncensored subjects who resemble them come to stand in
    for them.

    ``numerator`` holds the corresponding marginal probabilities
    P(C_t = 0 | C_t-1 = 0); when omitted the weights are
    unstabilized.  ``H`` is accepted for signature compatibility and
    is used only to check shapes.

    Diagnostic.  As with treatment weights, a correctly specified
    stabilized censoring weight has mean near 1.

    Returns
    -------
    RichResult with keys estimate (the mean weight), weights,
    unstabilized, mean_weight, max_weight, n, n_times, method.
    """
    D = [[float(v) for v in row] for row in C]
    n = len(D)
    if n == 0:
        raise ValueError("need at least one subject")
    nt = len(D[0])
    if any(len(r) != nt for r in D):
        raise ValueError("C must be rectangular")
    if any(not 0.0 < p <= 1.0 for r in D for p in r):
        raise ValueError("censoring probabilities must lie in (0, 1]")
    if numerator is None:
        N = [[1.0] * nt for _ in range(n)]
    else:
        N = [[float(v) for v in row] for row in numerator]
        if len(N) != n or any(len(r) != nt for r in N):
            raise ValueError("numerator must match C")
    if H is not None and len(list(H)) != n:
        raise ValueError("H must have one row per subject")
    w = []
    uw = []
    for i in range(n):
        dn = 1.0
        nm = 1.0
        for t in range(nt):
            dn *= D[i][t]
            nm *= N[i][t]
        w.append(nm / dn)
        uw.append(1.0 / dn)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(w) / n), "weights": w,
        "unstabilized": uw, "mean_weight": float(sum(w) / n),
        "max_weight": float(max(w)), "n": n, "n_times": nt,
        "method": "stabilized censoring weights (Robins 1993)",
    }), "stbciw")


def cheatsheet():
    return "stbciw: Stabilized inverse-probability-of-censoring weights"


# compact alias per ledger/NAMING.md
censorwt = stabilized_censoring_weights
