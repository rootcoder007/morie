"""Deprecated alias for :func:`morie.fn.bayes_simple`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bayes_simple import bayes_simple as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_51"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_51(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bayes_simple` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_51() is the book-coordinate name for bayes_simple(); "
        "it will be removed. Use morie.fn.bayes_simple() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
