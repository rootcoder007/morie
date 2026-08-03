"""Deprecated alias for :func:`morie.fn.treatment_b_randomized`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .treatment_b_randomized import treatment_b_randomized as _impl

__all__ = ["ca_chapter_9_equation_2"]


def ca_chapter_9_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.treatment_b_randomized` instead."""
    warnings.warn(
        "ca_chapter_9_equation_2() is the book-coordinate name for treatment_b_randomized(); "
        "it will be removed. Use morie.fn.treatment_b_randomized() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
