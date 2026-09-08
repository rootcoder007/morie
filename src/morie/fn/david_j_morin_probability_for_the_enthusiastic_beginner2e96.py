"""Deprecated alias for :func:`morie.fn.at_least_one_of_iid`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .at_least_one_of_iid import at_least_one_of_iid as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_96"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_96(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.at_least_one_of_iid` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_96() is the book-coordinate name for at_least_one_of_iid(); "
        "it will be removed. Use morie.fn.at_least_one_of_iid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
