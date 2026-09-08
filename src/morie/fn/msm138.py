# morie.fn -- function file (rootcoder007/morie)
"""Bayesian kernel BLUP.

Implements eq. (8.8) pp.281-282 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_8", "mvsml_bayesian_kernel_blup"]


def mvsml_categorical_count_eq_8_8(y, K, sigma2_u=1.0, sigma2_e=1.0, gibbs=True,
         n_iter=1200, burn_in=300, seed=42):
    """y = 1 mu + u + e with u ~ N(0, sigma2_u K) and
    e ~ N(0, sigma2_e I) (eq. 8.8): kernel ridge regression cast in a
    Bayesian framework, lambda = sigma2_e/sigma2_u.  The conditional
    mode of u is (sigma_u^-2 K^-1 + sigma_e^-2 I)^-1 sigma_e^-2
    (y - 1 mu), which is exactly Henderson's BLUP -- so with K the
    genomic relationship matrix this model IS GBLUP (p.282).  The
    kernel trick turns a large-p problem into an n-dimensional one.
    Keys: estimate."""
    f = _gp.bayesian_kernel_blup(y, K, sigma2_u=sigma2_u,
                                 sigma2_e=sigma2_e, gibbs=gibbs,
                                 n_iter=n_iter, burn_in=burn_in,
                                 seed=seed)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "u": f["u"],
                              "sigma2_u": f["sigma2_u"],
                              "sigma2_e": f["sigma2_e"],
                              "method": "Bayesian kernel BLUP (MVSML 2022 eq. 8.8)"})
    return with_describe_pointer(res, "msm138")


mvsml_bayesian_kernel_blup = mvsml_categorical_count_eq_8_8


def cheatsheet():
    return "msm138: Bayesian kernel BLUP"
