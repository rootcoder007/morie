# morie.fn -- function file (rootcoder007/morie)
"""Least squares basis coefficients.

Implements eq. (14.7)-(14.8) p.581 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_convolutional_nn_eq_14_7", "mvsml_fda_coefficients"]


def mvsml_convolutional_nn_eq_14_7(t, x_t, L2=5, kind="fourier"):
    """c-hat_i = (Psi'Psi)^-1 Psi' x_i(t) (eq. 14.7), with Psi the
    m x L2 basis matrix of eq. (14.8) evaluated at the observation
    times.  This least squares solution coincides with the maximum
    likelihood estimate. Keys: estimate."""
    Psi = _gp.fda_basis_matrix(t, L2, kind=kind)
    c = _gp.fda_basis_coefficients(Psi, x_t)
    res = RichResult(payload={"estimate": c[0], "c": c, "Psi": Psi,
                              "method": "basis coefficients (MVSML 2022 eq. 14.7-14.8)"})
    return with_describe_pointer(res, "msm270")


mvsml_fda_coefficients = mvsml_convolutional_nn_eq_14_7


def cheatsheet():
    return "msm270: Least squares basis coefficients"
