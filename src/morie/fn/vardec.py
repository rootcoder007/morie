# morie.fn -- function file (rootcoder007/morie)
"""VAR forecast error variance decomposition -- alias of :mod:`morie.fn.fevdc`.

DUPLICATE, resolved by aliasing (wave-2 DUPMAP: vardec -> fevdc).  Both
names denote the same quantity: with P the lower Cholesky factor of
Sigma_u and Theta_s the MA coefficient matrices of the fitted VAR, the
share of the h-step forecast error variance of variable i due to the
orthogonalised shock j is

    sum_{s=0}^{h} (Theta_s P)[i, j]^2  /  sum_j sum_{s=0}^{h} (Theta_s P)[i, j]^2

(Lutkepohl 2005, doi:10.1007/978-3-540-27752-1).  ``morie.fn.fevdc``
already implements it; a second copy would double the surface under a
name that reads right, so this module re-exports the one implementation.
"""

from __future__ import annotations

from .fevdc import fevd as var_variance_decomp

__all__ = ["var_variance_decomp"]


def cheatsheet():
    return "vardec: VAR FEVD -- alias of fevdc (Lutkepohl 2005)"
