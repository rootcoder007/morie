# morie.fn -- function file (rootcoder007/morie)
"""Extended kernel BLUP with genotype-by-environment interaction.

Implements eq. (8.10) pp.283-285 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries the previous chapter's topic label;
chapter 8 is Reproducing Kernel Hilbert Spaces regression, and the
canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_categorical_count_eq_8_10", "mvsml_kernel_blup_gxe"]


def mvsml_categorical_count_eq_8_10(Z_u1, K, Z_E, sigma2_u1=1.0, sigma2_u2=1.0):
    """y = mu 1 + Z_E beta_E + u_1 + u_2 + eps (eq. 8.10) with
    u_1 ~ N(0, sigma2_u1 K_1), K_1 = Z_u1 K Z_u1' the genomic main
    effects, and u_2 ~ N(0, sigma2_u2 K_2),
    K_2 = (Z_u1 K Z_u1') o (Z_E Z_E') the genotype-by-environment
    interaction, where "o" is the Hadamard product (p.285).
    Keys: estimate."""
    f = _gp.kernel_blup_gxe(Z_u1, K, Z_E, sigma2_u1=sigma2_u1,
                            sigma2_u2=sigma2_u2)
    ok1, _ = _gp.is_positive_semidefinite(f["K1"])
    ok2, _ = _gp.is_positive_semidefinite(f["K2"])
    res = RichResult(payload={"estimate": f["K1"][0][0],
                              "K1": f["K1"], "K2": f["K2"],
                              "K_env": f["K_env"],
                              "K1_psd": ok1, "K2_psd": ok2,
                              "method": "extended kernel BLUP with G x E (MVSML 2022 eq. 8.10)"})
    return with_describe_pointer(res, "msm144")


mvsml_kernel_blup_gxe = mvsml_categorical_count_eq_8_10


def cheatsheet():
    return "msm144: Extended kernel BLUP with genotype-by-environment interaction"
