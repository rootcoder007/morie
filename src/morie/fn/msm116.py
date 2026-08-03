# morie.fn -- function file (rootcoder007/morie)
"""Block-coordinate update for the multinomial model.

Implements eq. (7.9) p.227 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_9"]


def mvsml_bayesian_regression_pt2_eq_7_9(X, y, beta0, beta, lam=1.0, cls=0, baseline_last=True):
    """beta*_c = (X*'W_c X* + lambda D)^-1 X*'W_c y* (eq. 7.9), the
    block update of class c against a second-order Taylor
    approximation of the log-likelihood, with working response
    y*_ic = eta_ic + w_ic^-1(1{y_i = c} - p-tilde_c(x_i)) and
    weights w_ic = p-tilde_c(1 - p-tilde_c).  D is the identity with a
    zero first entry, so the intercept is unpenalized.
    Keys: estimate."""
    f = _gp.multinomial_block_update(X, y, beta0, beta, lam, cls,
                                     baseline_last=baseline_last)
    res = RichResult(payload={"estimate": f["beta0"],
                              "beta0": f["beta0"],
                              "beta": f["beta"],
                              "weights": f["weights"],
                              "working_response": f["working_response"],
                              "method": "multinomial block update (MVSML 2022 eq. 7.9)"})
    return with_describe_pointer(res, "msm116")


def cheatsheet():
    return "msm116: Block-coordinate update for the multinomial model"
