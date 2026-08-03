"""Deprecated alias for :func:`morie.fn.multinomial_probabilities`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multinomial_probabilities import multinomial_probabilities as _impl

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.multinomial_probabilities` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_pt2_eq_7_6() is the book-coordinate name for multinomial_probabilities(); "
        "it will be removed. Use morie.fn.multinomial_probabilities() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
