# morie.fn -- function file (rootcoder007/morie)
"""Matrix-variate form of the multi-trait model.

Implements eq. (6.9) pp.191-193 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(Y, Z1, G, X=None, n_iter=1200, burn_in=300, seed=42):
    """Y = 1_J mu' + X B + Z_1 b_1 + E (eq. 6.9) with
    E ~ MN(0, I_J, R) and b_1 ~ MN(0, G, Sigma_T), fitted by the
    Gibbs sampler of p.193.  The book notes that when Sigma_T and R
    are diagonal this is equivalent to fitting a univariate GBLUP to
    each trait separately (p.191). Keys: estimate."""
    f = _gp.multitrait_bayes_gibbs(Y, Z1, G, X=X, n_iter=n_iter,
                                   burn_in=burn_in, seed=seed)
    res = RichResult(payload={"estimate": f["mu"][0], "mu": f["mu"],
                              "b1": f["b1"], "Sigma_T": f["Sigma_T"],
                              "R": f["R"], "n_kept": f["n_kept"],
                              "method": "matrix-variate multi-trait model (MVSML 2022 eq. 6.9)"})
    return with_describe_pointer(res, "msm074")


def cheatsheet():
    return "msm074: Matrix-variate form of the multi-trait model"
