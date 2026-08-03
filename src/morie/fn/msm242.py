"""Deprecated alias for :func:`morie.fn.gblup_gebv`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gblup_gebv import gblup_gebv as _impl

__all__ = ["mvsml_preprocessing_eq_2_3"]


def mvsml_preprocessing_eq_2_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gblup_gebv` instead."""
    warnings.warn(
        "mvsml_preprocessing_eq_2_3() is the book-coordinate name for gblup_gebv(); "
        "it will be removed. Use morie.fn.gblup_gebv() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
