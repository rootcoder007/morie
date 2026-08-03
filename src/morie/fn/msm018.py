"""Deprecated alias for :func:`morie.fn.gxe_blup_model`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gxe_blup_model import gxe_blup_model as _impl

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def mvsml_linear_mixed_models_eq_5_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gxe_blup_model` instead."""
    warnings.warn(
        "mvsml_linear_mixed_models_eq_5_4() is the book-coordinate name for gxe_blup_model(); "
        "it will be removed. Use morie.fn.gxe_blup_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
