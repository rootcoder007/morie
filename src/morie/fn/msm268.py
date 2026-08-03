# morie.fn -- function file (rootcoder007/morie)
"""Residual variance of the functional fit.

Implements eq. (14.5) p.580 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_convolutional_nn_eq_14_5", "mvsml_fda_sigma2"]


def mvsml_convolutional_nn_eq_14_5(t, X_curves, y, L1=3, L2=5, kind="fourier"):
    """sigma2-hat = (1/n)(y - X* beta-hat)'(y - X* beta-hat)
    (eq. 14.5), the maximum likelihood variance -- divided by n, not
    by the residual degrees of freedom. Keys: estimate."""
    f = _gp.fda_fit(t, X_curves, y, L1=L1, L2=L2, kind=kind)
    res = RichResult(payload={"estimate": f["sigma2"],
                              "sigma2": f["sigma2"],
                              "residuals": f["residuals"],
                              "method": "functional residual variance (MVSML 2022 eq. 14.5)"})
    return with_describe_pointer(res, "msm268")


mvsml_fda_sigma2 = mvsml_convolutional_nn_eq_14_5


def cheatsheet():
    return "msm268: Residual variance of the functional fit"
