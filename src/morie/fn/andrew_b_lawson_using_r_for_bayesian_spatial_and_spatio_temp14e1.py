# morie.fn -- function file (rootcoder007/morie)
"""Shared-factor multivariate disease mean."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["mvfacmu", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_equation_1"]


def mvfacmu(e, lam, f):
    """Shared-factor multivariate disease mean.

    mu_ik = e_ik rho_ik,  log(rho_ik) = lambda_k f_i   (Lawson eq. 14.1).

    Shared spatial factor across k diseases: one common factor f_i per
    area, one loading lambda_k per disease.  ``e[i][k]`` are expected
    counts; the returned matrices are row-major, area by disease.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Shared-factor multivariate disease mean", payload=_c.mvfacmu(e=e, lam=lam, f=f))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_equation_1 = mvfacmu


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14e1: Shared-factor multivariate disease mean"
