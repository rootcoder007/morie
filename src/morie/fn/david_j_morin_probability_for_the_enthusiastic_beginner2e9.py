"""Deprecated alias for :func:`morie.fn.chain_rule`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .chain_rule import chain_rule as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.chain_rule` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_9() is the book-coordinate name for chain_rule(); "
        "it will be removed. Use morie.fn.chain_rule() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
