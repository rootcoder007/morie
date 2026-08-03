"""Deprecated alias for :func:`morie.fn.penalized_poisson_fit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .penalized_poisson_fit import penalized_poisson_fit as _impl

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_11"]


def mvsml_bayesian_regression_pt2_eq_7_11(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.penalized_poisson_fit` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_pt2_eq_7_11() is the book-coordinate name for penalized_poisson_fit(); "
        "it will be removed. Use morie.fn.penalized_poisson_fit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
