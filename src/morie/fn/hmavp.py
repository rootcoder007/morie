# morie.fn -- function file (rootcoder007/morie)
"""Global and windowed average pooling."""

from . import _array_core as np

from ._richresult import RichResult
from .grapl import geron_average_pooling_2d

__all__ = ["average_pooling", "geron_average_pool"]


def average_pooling(x, pool_size=2, stride=None, padding="valid",
                    global_pool=False):
    r"""Average pooling, windowed or global.

    ``global_pool=True`` collapses each channel to its mean over all
    spatial positions. That is the operation that replaced the flatten
    plus dense layer at the head of modern convolutional networks: it
    has NO parameters, so it cannot overfit the way a dense layer on a
    flattened feature map does, and it forces each channel to stand for
    a concept on its own rather than relying on a downstream mixing
    layer.

    Parameters
    ----------
    x : array-like
        ``(h, w)``, ``(h, w, c)`` or ``(n, h, w, c)``.
    pool_size, stride, padding
        As in :func:`~morie.fn.grapl.geron_average_pooling_2d`.
    global_pool : bool
        Average over all spatial positions instead.

    Returns
    -------
    RichResult
        ``pooled``, ``output_shape``, ``parameters`` (always 0).

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 12.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(16.0).reshape(4, 4)
    >>> float(average_pooling(a, global_pool=True)["pooled"])
    7.5
    """
    a = np.asarray(x, dtype=float)
    if not global_pool:
        out = geron_average_pooling_2d(a, pool_size, stride, padding)
        return RichResult(
            payload={
                "estimate": out["pooled"],
                "pooled": out["pooled"],
                "output_shape": out["output_shape"],
                "global": False,
                "parameters": 0,
                "method": "Windowed average pooling",
            }
        )
    if a.ndim == 2:
        res = float(a.mean())
    elif a.ndim == 3:
        res = a.mean(axis=(0, 1))
    elif a.ndim == 4:
        res = a.mean(axis=(1, 2))
    else:
        raise ValueError(
            "x must have 2, 3 or 4 dimensions, got %d." % a.ndim
        )
    return RichResult(
        payload={
            "estimate": res,
            "pooled": res,
            "output_shape": tuple(int(v) for v in np.shape(res)),
            "global": True,
            "parameters": 0,
            "note": (
                "global average pooling has no parameters, so unlike a dense "
                "head on a flattened map it cannot overfit the spatial "
                "layout; each channel has to mean something by itself"
            ),
            "method": "Global average pooling",
        }
    )


def cheatsheet():
    return "hmavp: windowed or global average pooling, the latter parameter-free"


#: Catalogue alias for :func:`average_pooling`.
geron_average_pool = average_pooling
