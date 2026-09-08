"""Deprecated alias for :func:`morie.fn.multitrait_ridge_form`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multitrait_ridge_form import multitrait_ridge_form as _impl

__all__ = ["mvsml_bayesian_regression_eq_6_10"]


def mvsml_bayesian_regression_eq_6_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.multitrait_ridge_form` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_eq_6_10() is the book-coordinate name for multitrait_ridge_form(); "
        "it will be removed. Use morie.fn.multitrait_ridge_form() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
