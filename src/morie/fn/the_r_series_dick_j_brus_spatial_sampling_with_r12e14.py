"""Deprecated alias for :func:`morie.fn.n_design_effect`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .n_design_effect import n_design_effect as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_14"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_14(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.n_design_effect` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_14() is the book-coordinate name for n_design_effect(); "
        "it will be removed. Use morie.fn.n_design_effect() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
