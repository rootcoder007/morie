"""Deprecated alias for :func:`morie.fn.suit_full_house_probability`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .suit_full_house_probability import suit_full_house_probability as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_41"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_41(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.suit_full_house_probability` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_41() is the book-coordinate name for suit_full_house_probability(); "
        "it will be removed. Use morie.fn.suit_full_house_probability() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
