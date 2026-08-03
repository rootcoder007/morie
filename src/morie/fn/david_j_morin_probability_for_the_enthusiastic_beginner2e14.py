"""Deprecated alias for :func:`morie.fn.prob_or_exclusive`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .prob_or_exclusive import prob_or_exclusive as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_14(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.prob_or_exclusive` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_14() is the book-coordinate name for prob_or_exclusive(); "
        "it will be removed. Use morie.fn.prob_or_exclusive() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
