"""Deprecated alias for :func:`morie.fn.psm_standardized_bias`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .psm_standardized_bias import psm_standardized_bias as _impl

__all__ = ["ca_chapter_10_equation_1"]


def ca_chapter_10_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.psm_standardized_bias` instead."""
    warnings.warn(
        "ca_chapter_10_equation_1() is the book-coordinate name for psm_standardized_bias(); "
        "it will be removed. Use morie.fn.psm_standardized_bias() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
