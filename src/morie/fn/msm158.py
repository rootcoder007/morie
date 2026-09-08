# morie.fn -- function file (rootcoder007/morie)

"""Extended approximate (compressed) kernel model.

Implements eq. (8.13) p.296 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["apxkern", "mvsml_categorical_count_eq_8_13"]


def apxkern(X, m_index, Z_u1, Z_E, kernel='linear', gamma=None):

    """y = mu 1 + Z_E beta_E + P_u1 f + P_u2 l + eps (eq. 8.13), the
    approximate-kernel model extended to several environments.  Steps
    1-7 of the summary on p.296 build P = K_{L,m} U S^(-1/2) from m of
    the L lines, expand it to the n records as P_u1 = Z_u1 P, and form
    the environment interaction P_u2 = P_u1 : Z_E, where ":" is the
    row-wise Kronecker product; step 8 fits the stacked design under
    ridge.  Keys: P, P_u1, P_u2, design, widths, rank.
    """

    res = RichResult(payload=_gp.approx_kernel_extended(X, m_index, Z_u1, Z_E, kernel=kernel,
                                 gamma=gamma))

    return with_describe_pointer(res, "msm158")


mvsml_categorical_count_eq_8_13 = apxkern


def cheatsheet():
    return "msm158: Extended approximate (compressed) kernel model"
