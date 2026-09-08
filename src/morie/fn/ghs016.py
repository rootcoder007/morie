"""Deprecated alias for :func:`morie.fn.cdp_posterior_cov`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cdp_posterior_cov import cdp_posterior_cov as _impl

__all__ = ["ghosal_ch3_dirichlet_posterior_cov"]


def ghosal_ch3_dirichlet_posterior_cov(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cdp_posterior_cov` instead."""
    warnings.warn(
        "ghosal_ch3_dirichlet_posterior_cov() is the book-coordinate name for cdp_posterior_cov(); "
        "it will be removed. Use morie.fn.cdp_posterior_cov() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
