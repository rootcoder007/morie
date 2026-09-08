"""Deprecated alias for :func:`morie.fn.rkhs_covariances`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .rkhs_covariances import rkhs_covariances as _impl

__all__ = ["mvsml_bayesian_regression_eq_6_7"]


def mvsml_bayesian_regression_eq_6_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.rkhs_covariances` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_eq_6_7() is the book-coordinate name for rkhs_covariances(); "
        "it will be removed. Use morie.fn.rkhs_covariances() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
