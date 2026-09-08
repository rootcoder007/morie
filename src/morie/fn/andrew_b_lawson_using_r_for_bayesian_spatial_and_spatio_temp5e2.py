# morie.fn -- function file (rootcoder007/morie)
"""Posterior-averaged Poisson residual."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["postres", "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2"]


def postres(y, e, theta_draws):
    """Posterior-averaged Poisson residual.

    r_i = y_i - (1/G) sum_g e_i theta_i^(g)   (Lawson eq. 5.2).

    Posterior-averaged Poisson residual for tract counts with expected
    count e_i.  ``theta_draws[g][i]`` is draw g of the relative risk for
    tract i; the average is taken over the posterior sample, which is
    what distinguishes (5.2) from a plug-in residual.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Posterior-averaged Poisson residual", payload=_c.postres(y=y, e=e, theta_draws=theta_draws))


andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2 = postres


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2: Posterior-averaged Poisson residual"
