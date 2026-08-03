"""Deprecated alias for :func:`morie.fn.excess_score_factor`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .excess_score_factor import excess_score_factor as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_81"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_81(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.excess_score_factor` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_81() is the book-coordinate name for excess_score_factor(); "
        "it will be removed. Use morie.fn.excess_score_factor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
