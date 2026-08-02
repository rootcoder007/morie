# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-padding around an input for valid/same convolutions."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_padding"]


def geron_padding(x, pad_h=None, pad_w=None, kernel_size=None, stride=1):
    """
    Zero-padding around input for valid/same convolutions.

    Formula: pad_h = (kh - 1)/2 for same padding

    "Valid" padding drops the border, shrinking the map by k-1 each
    convolution, so a deep stack eats its own edges. "Same" padding
    restores the loss with (k-1)/2 zeros a side -- exact only for ODD
    kernels; for an even kernel the split is asymmetric and the extra row
    is added at the bottom/right, which this function does explicitly
    rather than silently.

    Give ``kernel_size`` to have the same-padding computed, or give
    ``pad_h``/``pad_w`` directly.

    Parameters
    ----------
    x : array-like, shape (h, w), (h, w, c) or (n, h, w, c)
        Input map; padding is applied to the spatial axes only.
    pad_h, pad_w : int or (int, int), optional
        Explicit padding, symmetric or (before, after).
    kernel_size : int or (int, int), optional
        Compute same-padding from the kernel instead.
    stride : int, default 1
        Used only to report the resulting output size.

    Returns
    -------
    result : RichResult
        Keys: padded, pad_h, pad_w, output_shape, estimate, n, method.

    Examples
    --------
    >>> r = geron_padding([[1.0]], 1, 1)
    >>> r["padded"].shape
    (3, 3)
    >>> float(r["padded"].sum())
    1.0
    >>> float(r["padded"][1, 1])
    1.0

    A 3x3 kernel needs one zero a side to keep the map the same size:

    >>> r2 = geron_padding(np.ones((5, 5)), kernel_size=3)
    >>> r2["pad_h"], r2["output_shape"]
    ((1, 1), (5, 5))

    An even kernel cannot be split evenly; the extra row goes last:

    >>> geron_padding(np.ones((4, 4)), kernel_size=2)["pad_h"]
    (0, 1)

    References
    ----------
    Geron Ch 12
    """
    a = np.asarray(x, dtype=float)
    if a.ndim not in (2, 3, 4):
        raise ValueError(f"geron_padding: x must have 2, 3 or 4 dimensions, got ndim={a.ndim}")
    hax = 0 if a.ndim in (2, 3) else 1
    H, W = a.shape[hax], a.shape[hax + 1]

    def _pair(v, name):
        if np.ndim(v) == 0:
            k = int(v)
            if k < 0:
                raise ValueError(f"geron_padding: {name} must be non-negative, got {k}")
            return (k, k)
        t = tuple(int(u) for u in v)
        if len(t) != 2 or min(t) < 0:
            raise ValueError(f"geron_padding: {name} must be a non-negative int or a pair of them, got {v!r}")
        return t

    if kernel_size is not None:
        kh, kw = _pair(kernel_size, "kernel_size")
        if kh < 1 or kw < 1:
            raise ValueError(f"geron_padding: kernel_size must be >= 1, got {kernel_size!r}")
        ph = ((kh - 1) // 2, kh - 1 - (kh - 1) // 2)
        pw = ((kw - 1) // 2, kw - 1 - (kw - 1) // 2)
        if pad_h is not None or pad_w is not None:
            raise ValueError("geron_padding: give kernel_size or explicit pad_h/pad_w, not both")
    else:
        if pad_h is None and pad_w is None:
            raise ValueError("geron_padding: give either kernel_size or pad_h/pad_w")
        ph = _pair(0 if pad_h is None else pad_h, "pad_h")
        pw = _pair(0 if pad_w is None else pad_w, "pad_w")

    widths = [(0, 0)] * a.ndim
    widths[hax] = ph
    widths[hax + 1] = pw
    out = np.pad(a, widths, mode="constant", constant_values=0.0)

    s = int(stride)
    if s < 1:
        raise ValueError(f"geron_padding: stride must be >= 1, got {stride}")
    if kernel_size is not None:
        oh = (H + ph[0] + ph[1] - kh) // s + 1
        ow = (W + pw[0] + pw[1] - kw) // s + 1
    else:
        oh, ow = out.shape[hax], out.shape[hax + 1]

    return RichResult(
        title="Zero padding",
        summary_lines=[("Input", (H, W)), ("Padded", (out.shape[hax], out.shape[hax + 1])), ("Output", (oh, ow))],
        interpretation="Same padding keeps the spatial size at stride 1; an even kernel pads asymmetrically.",
        payload={
            "padded": out,
            "pad_h": ph,
            "pad_w": pw,
            "output_shape": (int(oh), int(ow)),
            "estimate": out,
            "n": int(a.size),
            "method": "Zero padding with explicit or same-convolution widths",
        },
    )


def cheatsheet():
    return "hmpd: Zero-padding around input for valid/same convolutions"
