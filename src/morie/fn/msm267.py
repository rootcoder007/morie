# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood estimate of the basis coefficients.

Implements eq. (14.4)-(14.5) p.580 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_convolutional_nn_eq_14_4", "mvsml_fda_fit"]


def mvsml_convolutional_nn_eq_14_4(t, X_curves, y, L1=3, L2=5, kind="fourier"):
    """beta-hat = (X*'X*)^-1 X*'y (eq. 14.4) and
    sigma2-hat = (1/n)(y - X*beta-hat)'(y - X*beta-hat) (eq. 14.5),
    the maximum likelihood estimates once the functional covariate has
    been reduced to scalar scores. Keys: estimate."""
    f = _gp.fda_fit(t, X_curves, y, L1=L1, L2=L2, kind=kind)
    res = RichResult(payload={"estimate": f["beta"][0],
                              "beta": f["beta"],
                              "sigma2": f["sigma2"],
                              "fitted": f["fitted"],
                              "method": "functional regression ML fit (MVSML 2022 eq. 14.4-14.5)"})
    return with_describe_pointer(res, "msm267")


mvsml_fda_fit = mvsml_convolutional_nn_eq_14_4


def cheatsheet():
    return "msm267: Maximum likelihood estimate of the basis coefficients"


# compact alias per ledger/NAMING.md
mvsmlfdafit = mvsml_fda_fit
