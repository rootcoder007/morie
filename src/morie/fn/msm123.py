# morie.fn -- function file (rootcoder007/morie)
"""RKHS optimization problem.

Implements eq. (8.1) p.253 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the auto-generated stub name carries the topic label of the
previous chapter; chapter 8 is Reproducing Kernel Hilbert Spaces
regression, and the canonical name below reflects that.  Both names
resolve to the same function.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_1", "mvsml_rkhs_objective"]


def mvsml_categorical_count_eq_8_1(K, y, beta, eta0=0.0, lam=1.0, loss="squared"):
    """min over f in H of {(1/n) sum_i L(y_i, f(x_i))
    + lambda ||f||_H^2} (eq. 8.1): the penalized empirical risk in a
    reproducing kernel Hilbert space, where L is minus the conditional
    log-likelihood for the response type and ||f||_H^2 measures model
    complexity.  Evaluates the objective at a given (eta_0, beta).
    Keys: estimate."""
    f = _gp.rkhs_predict(K, beta, eta0)
    ys = _gp._flat(y)
    n = len(ys)
    if loss == "squared":
        emp = sum((a - b) ** 2 for a, b in zip(ys, f)) / n
    elif loss == "logistic":
        emp = sum(math.log(1.0 + math.exp(-(2 * a - 1) * b))
                  for a, b in zip(ys, f)) / n
    elif loss == "hinge":
        emp = sum(max(0.0, 1.0 - (2 * a - 1) * b)
                  for a, b in zip(ys, f)) / n
    else:
        raise ValueError("unknown loss: %s" % loss)
    norm = _gp.rkhs_norm(beta, K)
    obj = emp + float(lam) * norm
    res = RichResult(payload={"estimate": obj, "objective": obj,
                              "empirical_risk": emp,
                              "rkhs_norm2": norm,
                              "method": "RKHS penalized risk (MVSML 2022 eq. 8.1)"})
    return with_describe_pointer(res, "msm123")


mvsml_rkhs_objective = mvsml_categorical_count_eq_8_1


def cheatsheet():
    return "msm123: RKHS optimization problem"
