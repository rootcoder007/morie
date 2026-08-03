# morie.fn -- function file (rootcoder007/morie)
"""Representer theorem solution.

Implements eq. (8.2) p.254 of Montesinos López, Montesinos López & Crossa
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

__all__ = ["mvsml_categorical_count_eq_8_2", "mvsml_rkhs_representer"]


def mvsml_categorical_count_eq_8_2(K_new, beta, eta0=0.0, K_train=None):
    """f(x_i) = eta_0 + sum_j beta_j K(x_i, x_j) = eta_0 + k_i'beta
    (eq. 8.2): by the representer theorem (Wahba 1990) the solution of
    eq. (8.1) lies in the finite-dimensional span of the kernel
    functions evaluated at the training points, so the fit needs n
    coefficients rather than p.  With ``K_train`` the RKHS norm
    ||f||_H^2 = sum_ij beta_i beta_j K(x_i, x_j) is also returned.
    Keys: estimate."""
    pred = _gp.rkhs_predict(K_new, beta, eta0)
    norm = _gp.rkhs_norm(beta, K_train) if K_train is not None \
        else None
    res = RichResult(payload={"estimate": pred[0],
                              "prediction": pred,
                              "rkhs_norm2": norm,
                              "n_coefficients": len(_gp._flat(beta)),
                              "method": "representer theorem (MVSML 2022 eq. 8.2)"})
    return with_describe_pointer(res, "msm125")


mvsml_rkhs_representer = mvsml_categorical_count_eq_8_2


def cheatsheet():
    return "msm125: Representer theorem solution"
