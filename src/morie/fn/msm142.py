# morie.fn -- function file (rootcoder007/morie)
"""Kernel BLUP with replicated individuals.

Implements eq. (8.9) p.282 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_9", "mvsml_kernel_blup_replicated"]


def mvsml_categorical_count_eq_8_9(Z, K, sigma2_u=1.0):
    """Y = 1 mu + Z u + e (eq. 8.9), the form used when individuals
    have more than one replication.  The predictor cannot be handed to
    BGLR directly, so its covariance K_* = Var(Z u) = Z K Z' is
    precomputed and used as the kernel. Keys: estimate."""
    Ks = _gp.kernel_blup_replicated(Z, K, sigma2_u=sigma2_u)
    ok, _ = _gp.is_positive_semidefinite(Ks)
    res = RichResult(payload={"estimate": Ks[0][0], "K_star": Ks,
                              "positive_semidefinite": ok,
                              "method": "replicated kernel BLUP covariance (MVSML 2022 eq. 8.9)"})
    return with_describe_pointer(res, "msm142")


mvsml_kernel_blup_replicated = mvsml_categorical_count_eq_8_9


def cheatsheet():
    return "msm142: Kernel BLUP with replicated individuals"
