"""Deprecated alias for :func:`morie.fn.hockey_stick`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .hockey_stick import hockey_stick as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_29"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_29(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.hockey_stick` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_29() is the book-coordinate name for hockey_stick(); "
        "it will be removed. Use morie.fn.hockey_stick() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
