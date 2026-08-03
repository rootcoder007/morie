"""Deprecated alias for :func:`morie.fn.cdp_posterior_var`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cdp_posterior_var import cdp_posterior_var as _impl

__all__ = ["ghosal_ch3_dirichlet_posterior_var"]


def ghosal_ch3_dirichlet_posterior_var(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cdp_posterior_var` instead."""
    warnings.warn(
        "ghosal_ch3_dirichlet_posterior_var() is the book-coordinate name for cdp_posterior_var(); "
        "it will be removed. Use morie.fn.cdp_posterior_var() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
