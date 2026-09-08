"""Deprecated alias for :func:`morie.fn.fisher_information_reml`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .fisher_information_reml import fisher_information_reml as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.fisher_information_reml` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_2() is the book-coordinate name for fisher_information_reml(); "
        "it will be removed. Use morie.fn.fisher_information_reml() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
