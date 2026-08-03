"""Deprecated alias for :func:`morie.fn.ordinal_probit_gblup_gibbs`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ordinal_probit_gblup_gibbs import ordinal_probit_gblup_gibbs as _impl

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_4"]


def mvsml_bayesian_regression_pt2_eq_7_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ordinal_probit_gblup_gibbs` instead."""
    warnings.warn(
        "mvsml_bayesian_regression_pt2_eq_7_4() is the book-coordinate name for ordinal_probit_gblup_gibbs(); "
        "it will be removed. Use morie.fn.ordinal_probit_gblup_gibbs() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
