"""Deprecated alias for :func:`morie.fn.multinomial_loglik`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multinomial_loglik import multinomial_loglik as _impl

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_8"]


def mvsml_bayesian_regression_pt2_eq_7_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.multinomial_loglik` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_pt2_eq_7_8() is the book-coordinate name for multinomial_loglik(); "
        "it will be removed. Use morie.fn.multinomial_loglik() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
