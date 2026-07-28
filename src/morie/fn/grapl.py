# morie.fn -- function file (rootcoder007/morie)
"""2D average pooling."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_average_pooling_2d"]


def geron_average_pooling_2d(x, pool_size=2, stride=None, padding="valid"):
    r"""Average pooling over :math:`k \times k` windows.

    .. math:: y_{ij} = \frac{1}{k^2}\sum_{a,b} x_{si+a,\, sj+b}

    Average pooling and max pooling are not interchangeable. Max
    pooling keeps the strongest activation and discards the rest,
    which gives stronger translation invariance and is why it
    dominates in practice; average pooling keeps every input's
    contribution, so it loses less information and its gradient
    reaches every element of the window rather than one. That matters
    at the last layer, where global average pooling is the standard
    way to reduce a feature map to one number per channel without
    adding parameters.

    ``padding='same'`` pads with edge values so the output keeps the
    input size under unit stride; zero padding would drag borders
    toward zero, which is a bias rather than a boundary convention.

    Parameters
    ----------
    x : array-like, shape (h, w) or (h, w, c) or (n, h, w, c)
    pool_size : int or (int, int)
    stride : int or (int, int), optional
        Defaults to ``pool_size`` -- non-overlapping windows.
    padding : {'valid', 'same'}

    Returns
    -------
    RichResult
        ``pooled``, ``output_shape``, ``reduction``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 12,
    average pooling.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(16.0).reshape(4, 4)
    >>> out = geron_average_pooling_2d(a, 2)
    >>> out["pooled"].tolist()
    [[2.5, 4.5], [10.5, 12.5]]
    """
    a = np.asarray(x, dtype=float)
    squeeze_batch = squeeze_chan = False
    if a.ndim == 2:
        a = a[None, :, :, None]
        squeeze_batch = squeeze_chan = True
    elif a.ndim == 3:
        a = a[None, ...]
        squeeze_batch = True
    if a.ndim != 4:
        raise ValueError(
            "x must have 2, 3 or 4 dimensions, got %d." % np.ndim(x)
        )
    kh, kw = (pool_size, pool_size) if np.isscalar(pool_size) else pool_size
    kh, kw = int(kh), int(kw)
    if kh < 1 or kw < 1:
        raise ValueError("pool_size must be positive, got %r." % (pool_size,))
    st = pool_size if stride is None else stride
    sh, sw = (st, st) if np.isscalar(st) else st
    sh, sw = int(sh), int(sw)
    if sh < 1 or sw < 1:
        raise ValueError("stride must be positive, got %r." % (stride,))
    if padding not in ("valid", "same"):
        raise ValueError(
            "padding must be 'valid' or 'same', got %r." % padding
        )

    n, h, w, c = a.shape
    if padding == "same":
        oh = int(np.ceil(h / sh))
        ow = int(np.ceil(w / sw))
        ph = max((oh - 1) * sh + kh - h, 0)
        pw = max((ow - 1) * sw + kw - w, 0)
        a = np.pad(a, ((0, 0), (ph // 2, ph - ph // 2),
                       (pw // 2, pw - pw // 2), (0, 0)), mode="edge")
        h, w = a.shape[1], a.shape[2]
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    if oh < 1 or ow < 1:
        raise ValueError(
            "pool window %dx%d does not fit in a %dx%d input." % (kh, kw, h, w)
        )
    out = np.empty((n, oh, ow, c))
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = a[:, i * sh:i * sh + kh,
                                j * sw:j * sw + kw, :].mean(axis=(1, 2))
    res = out
    if squeeze_chan:
        res = res[..., 0]
    if squeeze_batch:
        res = res[0]
    return RichResult(
        payload={
            "estimate": res,
            "pooled": res,
            "output_shape": tuple(int(v) for v in np.shape(res)),
            "pool_size": (kh, kw),
            "stride": (sh, sw),
            "padding": padding,
            "reduction": float(kh * kw / (sh * sw)),
            "note": (
                "average pooling keeps every input's contribution and passes "
                "gradient to the whole window; max pooling keeps one and is "
                "the more translation-invariant of the two"
            ),
            "method": "2D average pooling",
        }
    )


def cheatsheet():
    return "grapl: 2D average pooling with valid/same padding"
