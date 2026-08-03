"""Deprecated alias for :func:`morie.fn.treatment_b_confounded`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .treatment_b_confounded import treatment_b_confounded as _impl

__all__ = ["ca_chapter_9_equation_1"]


def ca_chapter_9_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.treatment_b_confounded` instead."""
    warnings.warn(
        "ca_chapter_9_equation_1() is the book-coordinate name for treatment_b_confounded(); "
        "it will be removed. Use morie.fn.treatment_b_confounded() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
