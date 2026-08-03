"""Deprecated alias for :func:`morie.fn.piecewise_cubic`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .piecewise_cubic import piecewise_cubic as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_34"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_34(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.piecewise_cubic` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_34() is the book-coordinate name for piecewise_cubic(); "
        "it will be removed. Use morie.fn.piecewise_cubic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
