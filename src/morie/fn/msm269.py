# morie.fn -- function file (rootcoder007/morie)
"""Basis expansion of the covariate curve.

Implements eq. (14.6) p.581 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_convolutional_nn_eq_14_6", "mvsml_fda_curve_expansion"]


def mvsml_convolutional_nn_eq_14_6(t, c, L2=None, kind="fourier"):
    """x_i(t) = sum_o c_io psi_o(t) (eq. 14.6): the functional
    covariate is observed only at finitely many points, so it is
    written as a linear combination of L2 basis functions.
    Keys: estimate."""
    cs = _gp._flat(c)
    L = len(cs) if L2 is None else int(L2)
    Psi = _gp.fda_basis_matrix(t, L, kind=kind)
    vals = [sum(Psi[j][o] * cs[o] for o in range(L))
            for j in range(len(Psi))]
    res = RichResult(payload={"estimate": vals[0], "x_t": vals,
                              "Psi": Psi,
                              "method": "curve basis expansion (MVSML 2022 eq. 14.6)"})
    return with_describe_pointer(res, "msm269")


mvsml_fda_curve_expansion = mvsml_convolutional_nn_eq_14_6


def cheatsheet():
    return "msm269: Basis expansion of the covariate curve"
