"""Deprecated alias for :func:`morie.fn.e_x_squared`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .e_x_squared import e_x_squared as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_70"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_70(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.e_x_squared` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_70() is the book-coordinate name for e_x_squared(); "
        "it will be removed. Use morie.fn.e_x_squared() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
