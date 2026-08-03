"""Deprecated alias for :func:`morie.fn.cdp_posterior_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cdp_posterior_mean import cdp_posterior_mean as _impl

__all__ = ["ghosal_ch3_dirichlet_posterior_mean"]


def ghosal_ch3_dirichlet_posterior_mean(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cdp_posterior_mean` instead."""
    warnings.warn(
        "ghosal_ch3_dirichlet_posterior_mean() is the book-coordinate name for cdp_posterior_mean(); "
        "it will be removed. Use morie.fn.cdp_posterior_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
