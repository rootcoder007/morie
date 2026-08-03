"""Deprecated alias for :func:`morie.fn.gblup_model`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gblup_model import gblup_model as _impl

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gblup_model` instead."""
    warnings.warn(
        "mvsml_linear_mixed_models_eq_5_3() is the book-coordinate name for gblup_model(); "
        "it will be removed. Use morie.fn.gblup_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
