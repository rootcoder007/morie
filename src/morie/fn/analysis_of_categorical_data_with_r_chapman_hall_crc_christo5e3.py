"""Deprecated alias for :func:`morie.fn.model_averaged_estimate`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .model_averaged_estimate import model_averaged_estimate as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.model_averaged_estimate` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_3() is the book-coordinate name for model_averaged_estimate(); "
        "it will be removed. Use morie.fn.model_averaged_estimate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
