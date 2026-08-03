"""Deprecated alias for :func:`morie.fn.bmtme_conditionals`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bmtme_conditionals import bmtme_conditionals as _impl

__all__ = ["mvsml_bayesian_regression_eq_6_11"]


def mvsml_bayesian_regression_eq_6_11(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bmtme_conditionals` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_eq_6_11() is the book-coordinate name for bmtme_conditionals(); "
        "it will be removed. Use morie.fn.bmtme_conditionals() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
