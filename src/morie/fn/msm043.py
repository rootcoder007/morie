# morie.fn -- function file (rootcoder007/morie)
"""Non-informative prior for the linear model.

Implements eq. (6.2) p.172 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_2"]


def mvsml_bayesian_regression_eq_6_2(sigma2, beta=None):
    """f(beta, sigma2) proportional to sigma^-2 (eq. 6.2): flat in
    beta and in log(sigma), improper because it does not integrate to
    one, yet yielding a proper posterior whenever X has full column
    rank (p.172).  Returns the prior density up to its (infinite)
    normalizing constant. Keys: estimate."""
    s2 = float(sigma2)
    if s2 <= 0:
        raise ValueError("sigma2 must be positive")
    dens = s2 ** -2
    res = RichResult(payload={"estimate": dens, "density": dens,
                              "log_density": -2.0 * math.log(s2),
                              "proper": False,
                              "method": "non-informative prior (MVSML 2022 eq. 6.2)"})
    return with_describe_pointer(res, "msm043")


def cheatsheet():
    return "msm043: Non-informative prior for the linear model"
