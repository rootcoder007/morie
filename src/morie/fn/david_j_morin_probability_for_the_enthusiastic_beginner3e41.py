"""Deprecated alias for :func:`morie.fn.sd_scale`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sd_scale import sd_scale as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_41"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_41(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.sd_scale` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_41() is the book-coordinate name for sd_scale(); "
        "it will be removed. Use morie.fn.sd_scale() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
