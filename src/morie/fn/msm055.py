# morie.fn -- function file (rootcoder007/morie)
"""GBLUP with an incidence matrix of genotypes.

Implements eq. (6.5) p.177 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_5"]


def mvsml_bayesian_regression_eq_6_5(y, Z, G, n_iter=2000, burn_in=500, seed=42, **kw):
    """Y = 1_n mu + Z g + eps (eq. 6.5), the form used when
    individuals are replicated.  BGLR cannot take this predictor
    directly, so the book precomputes the covariance of the predictor,
    K_L = Var(Z g) = Z G Z' (p.177), and fits an RKHS model on it;
    that is what happens here, via the Cholesky factor of K_L.
    Keys: estimate."""
    K = _gp.rkhs_covariances(Z, G)["K_L"]
    n = len(K)
    Kr = [[K[i][j] + (1e-8 if i == j else 0.0) for j in range(n)]
          for i in range(n)]
    f = _gp.bayes_gblup_gibbs(y, Kr, n_iter=n_iter, burn_in=burn_in,
                              seed=seed, **kw)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "g": f["g"], "K_L": K,
                              "sigma2": f["sigma2"],
                              "method": "GBLUP with genotype incidence matrix (MVSML 2022 eq. 6.5)"})
    return with_describe_pointer(res, "msm055")


def cheatsheet():
    return "msm055: GBLUP with an incidence matrix of genotypes"
