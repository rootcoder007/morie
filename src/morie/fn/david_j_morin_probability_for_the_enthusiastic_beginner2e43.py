"""Deprecated alias for :func:`morie.fn.at_most_two_suits_probability`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .at_most_two_suits_probability import at_most_two_suits_probability as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_43"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_43(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.at_most_two_suits_probability` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_43() is the book-coordinate name for at_most_two_suits_probability(); "
        "it will be removed. Use morie.fn.at_most_two_suits_probability() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
