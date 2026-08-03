"""Deprecated alias for :func:`morie.fn.blue_blup_via_v`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .blue_blup_via_v import blue_blup_via_v as _impl

__all__ = ["mvsml_preprocessing_eq_2_1"]


def mvsml_preprocessing_eq_2_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.blue_blup_via_v` instead."""
    warnings.warn(
        "mvsml_preprocessing_eq_2_1() is the book-coordinate name for blue_blup_via_v(); "
        "it will be removed. Use morie.fn.blue_blup_via_v() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
