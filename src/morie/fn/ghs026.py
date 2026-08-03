"""Deprecated alias for :func:`morie.fn._bits`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from ._bits import _bits as _impl

__all__ = ["ghosal_ch3_tailfree_finite_density_pm"]


def ghosal_ch3_tailfree_finite_density_pm(*args, **kwargs):
    """Deprecated; use :func:`morie.fn._bits` instead."""
    warnings.warn(
        "ghosal_ch3_tailfree_finite_density_pm() is the book-coordinate name for _bits(); "
        "it will be removed. Use morie.fn._bits() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
