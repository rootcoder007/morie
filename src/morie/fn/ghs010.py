"""Deprecated alias for :func:`morie.fn.discrete_hazard`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .discrete_hazard import discrete_hazard as _impl

__all__ = ["ghosal_ch3_discrete_hazard_rate"]


def ghosal_ch3_discrete_hazard_rate(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.discrete_hazard` instead."""
    warnings.warn(
        "ghosal_ch3_discrete_hazard_rate() is the book-coordinate name for discrete_hazard(); "
        "it will be removed. Use morie.fn.discrete_hazard() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
