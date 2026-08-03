"""Deprecated alias for :func:`morie.fn.factorial`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .factorial import factorial as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_1"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.factorial` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_1() is the book-coordinate name for factorial(); "
        "it will be removed. Use morie.fn.factorial() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
