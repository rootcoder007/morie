"""Deprecated alias for :func:`morie.fn.pt_set_mass_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pt_set_mass_mean import pt_set_mass_mean as _impl

__all__ = ["ghosal_ch3_polya_tree_first_two_moments"]


def ghosal_ch3_polya_tree_first_two_moments(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.pt_set_mass_mean` instead."""
    warnings.warn(
        "ghosal_ch3_polya_tree_first_two_moments() is the book-coordinate name for pt_set_mass_mean(); "
        "it will be removed. Use morie.fn.pt_set_mass_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
