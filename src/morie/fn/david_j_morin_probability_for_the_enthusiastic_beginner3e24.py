"""Deprecated alias for :func:`morie.fn.var_scale`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .var_scale import var_scale as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_24"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_24(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.var_scale` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_24() is the book-coordinate name for var_scale(); "
        "it will be removed. Use morie.fn.var_scale() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
