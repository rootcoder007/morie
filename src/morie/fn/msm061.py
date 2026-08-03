"""Deprecated alias for :func:`morie.fn.extended_predictor`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .extended_predictor import extended_predictor as _impl

__all__ = ["mvsml_bayesian_regression_eq_6_6"]


def mvsml_bayesian_regression_eq_6_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.extended_predictor` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_eq_6_6() is the book-coordinate name for extended_predictor(); "
        "it will be removed. Use morie.fn.extended_predictor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
