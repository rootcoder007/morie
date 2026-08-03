"""Deprecated alias for :func:`morie.fn.mme_solve`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .mme_solve import mme_solve as _impl

__all__ = ["mvsml_preprocessing_eq_2_2"]


def mvsml_preprocessing_eq_2_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.mme_solve` instead."""
    warnings.warn(
        "mvsml_preprocessing_eq_2_2() is the book-coordinate name for mme_solve(); "
        "it will be removed. Use morie.fn.mme_solve() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
