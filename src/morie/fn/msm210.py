# morie.fn -- function file (rootcoder007/morie)
"""Wolfe dual expansion.

Implements eq. (9.31) p.349 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_31", "mvsml_svm_dual_expansion"]


def mvsml_ridge_lasso_elastic_eq_9_31(alpha, X, y, beta0=0.0, K=None):
    """Substituting (9.28) and (9.29) back into the Lagrangian gives
    L(alpha) = (1/2)||sum_i alpha_i y_i x_i||^2
    - sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j)
    - sum_i alpha_i y_i beta_0 + sum_i alpha_i (eq. 9.31).  The middle
    term is -2x the first, and the third vanishes by (9.29), which is
    what collapses the expression to eq. (9.32). Keys: estimate."""
    a = _gp._flat(alpha)
    ys = _gp._flat(y)
    n = len(a)
    G = _gp._mat(K) if K is not None else \
        _gp._mm(_gp._mat(X), _gp._t(_gp._mat(X)))
    quad = sum(a[i] * a[j] * ys[i] * ys[j] * G[i][j]
               for i in range(n) for j in range(n))
    balance = sum(a[i] * ys[i] for i in range(n))
    val = 0.5 * quad - quad - balance * float(beta0) + sum(a)
    res = RichResult(payload={"estimate": val, "L": val,
                              "norm_term": 0.5 * quad,
                              "cross_term": -quad,
                              "balance_term": -balance * float(beta0),
                              "method": "Wolfe dual expansion (MVSML 2022 eq. 9.31)"})
    return with_describe_pointer(res, "msm210")


mvsml_svm_dual_expansion = mvsml_ridge_lasso_elastic_eq_9_31


def cheatsheet():
    return "msm210: Wolfe dual expansion"
