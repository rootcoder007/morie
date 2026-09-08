"""Deprecated alias for :func:`morie.fn.ridge_fit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ridge_fit import ridge_fit as _impl

__all__ = ["mvsml_elements_lin_reg_eq_3_5"]


def mvsml_elements_lin_reg_eq_3_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ridge_fit` instead."""
    warnings.warn(
        "mvsml_elements_lin_reg_eq_3_5() is the book-coordinate name for ridge_fit(); "
        "it will be removed. Use morie.fn.ridge_fit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
