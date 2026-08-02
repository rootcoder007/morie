# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conv layer output spatial size from input size, filter, padding, stride."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_conv_output_size"]

_METHOD = "Convolution output-size arithmetic"


def geron_conv_output_size(in_size, kernel, padding=0, stride=1, dilation=1):
    r"""Output spatial size of a convolution, per axis.

    .. math::
        \text{out} = \left\lfloor
        \frac{\text{in} + 2p - d(k-1) - 1}{s}\right\rfloor + 1

    which reduces to :math:`\lfloor (\text{in}+2p-k)/s \rfloor + 1` at
    dilation 1.  The floor is where input pixels go missing: when
    ``(in + 2p - k)`` is not a multiple of ``s`` the last few columns are
    never covered by any window, and ``dropped_cells`` reports how many.

    Parameters
    ----------
    in_size : int or sequence of int
        Input size along each spatial axis.
    kernel : int or sequence of int
        Filter size per axis.
    padding : int or sequence of int, optional
        Zero-padding added to *each* side, default 0.
    stride : int or sequence of int, optional
        Step per axis, default 1.
    dilation : int or sequence of int, optional
        Kernel dilation, default 1.

    Returns
    -------
    RichResult
        Payload keys ``out_size``, ``receptive_field``, ``same_padding``
        (padding that would preserve the size at stride 1),
        ``dropped_cells``, ``is_same``, ``estimate`` (first axis output
        size), ``n``, ``method``.

    References
    ----------
    Géron Ch 12, padding / stride output-size equation.

    Examples
    --------
    ``3x3`` filter with padding 1 at stride 1 is size-preserving:

    >>> r = geron_conv_output_size(28, 3, padding=1, stride=1)
    >>> r["out_size"]
    [28]
    >>> r["is_same"]
    True

    Halving with stride 2 and no padding loses a row:

    >>> r2 = geron_conv_output_size(28, 3, padding=0, stride=2)
    >>> r2["out_size"]
    [13]
    >>> r2["dropped_cells"]
    [1]

    Two axes at once:

    >>> geron_conv_output_size([32, 64], [5, 3], padding=[2, 1], stride=[1, 2])["out_size"]
    [32, 32]
    """
    def _vec(v, name, ndim=None):
        a = np.atleast_1d(np.asarray(v))
        if a.ndim != 1:
            raise ValueError(f"{name} must be a scalar or 1-D sequence, got ndim={a.ndim}.")
        try:
            ai = a.astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integers.") from exc
        if not np.array_equal(ai, a):
            raise ValueError(f"{name} must contain whole numbers, got {v!r}.")
        if ndim is not None and ai.size == 1:
            ai = np.repeat(ai, ndim)
        if ndim is not None and ai.size != ndim:
            raise ValueError(f"{name} must have 1 or {ndim} entries, got {ai.size}.")
        return ai

    ins = _vec(in_size, "in_size")
    nd = ins.size
    ks = _vec(kernel, "kernel", nd)
    ps = _vec(padding, "padding", nd)
    ss = _vec(stride, "stride", nd)
    ds = _vec(dilation, "dilation", nd)
    if np.any(ins < 1):
        raise ValueError(f"in_size must be positive, got {ins.tolist()}.")
    if np.any(ks < 1):
        raise ValueError(f"kernel must be positive, got {ks.tolist()}.")
    if np.any(ps < 0):
        raise ValueError(f"padding must be non-negative, got {ps.tolist()}.")
    if np.any(ss < 1):
        raise ValueError(f"stride must be positive, got {ss.tolist()}.")
    if np.any(ds < 1):
        raise ValueError(f"dilation must be positive, got {ds.tolist()}.")

    rf = ds * (ks - 1) + 1
    span = ins + 2 * ps - rf
    if np.any(span < 0):
        bad = np.flatnonzero(span < 0).tolist()
        raise ValueError(
            f"axes {bad}: the receptive field {rf[span < 0].tolist()} exceeds the "
            f"padded input {(ins + 2 * ps)[span < 0].tolist()}; output would be empty."
        )
    out = span // ss + 1
    dropped = span - (out - 1) * ss
    same_pad = (rf - 1) // 2

    return RichResult(
        title="Conv output size",
        summary_lines=[("Output", out.tolist()), ("Receptive field", rf.tolist())],
        payload={
            "out_size": out.tolist(),
            "receptive_field": rf.tolist(),
            "same_padding": same_pad.tolist(),
            "dropped_cells": dropped.tolist(),
            "is_same": bool(np.all(out == ins)),
            "estimate": float(out[0]),
            "n": int(nd),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grcos: conv output size = floor((in + 2p - d(k-1) - 1)/s) + 1"
