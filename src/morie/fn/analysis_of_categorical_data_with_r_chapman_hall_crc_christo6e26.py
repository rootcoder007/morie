"""Deprecated alias for :func:`morie.fn.group_testing_expected_tests`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .group_testing_expected_tests import group_testing_expected_tests as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_26"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_26(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.group_testing_expected_tests` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_26() is the book-coordinate name for group_testing_expected_tests(); "
        "it will be removed. Use morie.fn.group_testing_expected_tests() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
