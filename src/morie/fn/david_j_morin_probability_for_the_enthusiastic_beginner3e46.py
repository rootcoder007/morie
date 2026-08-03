"""Deprecated alias for :func:`morie.fn.sd_bernoulli`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sd_bernoulli import sd_bernoulli as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_46"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_46(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.sd_bernoulli` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_46() is the book-coordinate name for sd_bernoulli(); "
        "it will be removed. Use morie.fn.sd_bernoulli() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
