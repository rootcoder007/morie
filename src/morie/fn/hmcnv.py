# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convolutional layer forward pass.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 12 Equation 12-1, p. 423
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["convlayer", "geron_convolutional_layer"]

_METHOD = "Convolutional layer forward pass"


def convlayer(x, kernel, bias=None, stride=(1, 1), padding=(0, 0)):
    """Convolutional layer forward pass.

    Equation 12-1, p. 423 -- output of a convolutional layer.

    z[i, j, k] = b[k] + sum_u sum_v sum_k' x[i', j', k'] w[u, v, k', k]
    with i' = i * sh + u and j' = j * sw + v

    ``x`` is height by width by in-channels, ``kernel`` is fh by fw by
    in-channels by out-channels, both as nested lists.  ``padding`` is
    the zero padding named on p. 421.  This is a cross-correlation, as
    the book's own footnote 6 on p. 419 points out.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._geron.convlayer``.
    kernel : as documented for the shelf core
        See ``morie.fn._geron.convlayer``.
    bias : as documented for the shelf core
        See ``morie.fn._geron.convlayer``.
    stride : as documented for the shelf core
        See ``morie.fn._geron.convlayer``.
    padding : as documented for the shelf core
        See ``morie.fn._geron.convlayer``.

    Returns
    -------
    result : RichResult
        Payload keys: height, width, channels, total, nparams.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 12 Equation 12-1, p. 423
    """
    res = _core.convlayer(x=x, kernel=kernel, bias=bias, stride=stride, padding=padding)
    return RichResult(
        title=_METHOD,
        summary_lines=[("height", res["height"]), ("width", res["width"]), ("channels", res["channels"]), ("total", res["total"]), ("nparams", res["nparams"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_convolutional_layer = convlayer


def cheatsheet():
    return "convlayer: Convolutional layer forward pass"
