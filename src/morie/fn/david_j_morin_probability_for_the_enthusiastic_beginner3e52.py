"""Deprecated alias for :func:`morie.fn.sd_fair_coin_avg`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sd_fair_coin_avg import sd_fair_coin_avg as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_52"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_52(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.sd_fair_coin_avg` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_52() is the book-coordinate name for sd_fair_coin_avg(); "
        "it will be removed. Use morie.fn.sd_fair_coin_avg() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
