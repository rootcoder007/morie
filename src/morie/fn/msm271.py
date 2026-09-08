# morie.fn -- function file (rootcoder007/morie)

"""Basis matrix of the observed curves.

Implements eq. (14.8) p.581 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["basmat", "mvsml_convolutional_nn_eq_14_8"]


def basmat(t, n_basis, kind='fourier', period=None):

    """Psi is the m x L2 matrix whose (j, o) entry is psi_o(t_j)
    (eq. 14.8): rows are the times t_1, ..., t_m at which the
    covariate curve was actually observed and columns are the L2
    basis functions.  It is what turns the discretely sampled curve
    x_i(t) into basis coefficients, c-hat_i = (Psi'Psi)^-1 Psi' x_i(t)
    (eq. 14.7), and from there into the L1 functional scores that
    form the design matrix.  Keys: Psi, m, L2, PsiTPsi.
    """

    Psi = _gp.fda_basis_matrix(t, n_basis, kind=kind,
                               period=period)
    PtP = [[sum(Psi[r][i] * Psi[r][j] for r in range(len(Psi)))
            for j in range(len(Psi[0]))] for i in range(len(Psi[0]))]
    res = RichResult(payload={"Psi": Psi, "m": len(Psi),
                              "L2": len(Psi[0]), "PsiTPsi": PtP})

    return with_describe_pointer(res, "msm271")


mvsml_convolutional_nn_eq_14_8 = basmat


def cheatsheet():
    return "msm271: Basis matrix of the observed curves"
