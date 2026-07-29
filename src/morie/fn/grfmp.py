# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feature-map size of a convolutional layer."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_feature_map_dim"]

_METHOD = "Convolutional feature-map size"


def geron_feature_map_dim(H_out, W_out, C_out, bytes_per_value=4, batch_size=1):
    r"""How many numbers one conv layer's output holds.

    .. math::
        \dim = H' \times W' \times C_{\text{out}}

    Get ``H'`` and ``W'`` from
    :func:`morie.fn.grcos.geron_conv_output_size` -- the spatial
    arithmetic (padding, stride, dilation) lives there and is not
    repeated here.

    The reason to compute this at all is memory.  Training holds every
    layer's feature map for the backward pass, and the maps are
    per-*instance*, so the requirement scales with the batch.  A
    150x100x200 map is 3 MB in float32 for one image and 300 MB for a
    batch of 100 -- which is the usual explanation for an out-of-memory
    error that a smaller batch fixes.

    Parameters
    ----------
    H_out, W_out : int
        Output spatial size, positive.
    C_out : int
        Feature maps (filters), positive.
    bytes_per_value : int, optional
        4 for float32 (default), 2 for float16/bfloat16.
    batch_size : int, optional
        Instances held at once, default 1.

    Returns
    -------
    RichResult
        Payload keys ``dim``, ``bytes``, ``megabytes``, ``batch_bytes``,
        ``batch_megabytes``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 12, Feature Maps section.

    Examples
    --------
    A 28x28 map with 32 filters:

    >>> r = geron_feature_map_dim(28, 28, 32)
    >>> r["dim"]
    25088
    >>> r["bytes"]
    100352

    Géron's 150x100 map with 200 filters, in float32, for a batch of
    100 -- 1.2 GB:

    >>> r2 = geron_feature_map_dim(150, 100, 200, batch_size=100)
    >>> round(r2["megabytes"], 4)
    11.4441
    >>> round(r2["batch_megabytes"], 2)
    1144.41

    Half precision halves it exactly:

    >>> geron_feature_map_dim(150, 100, 200, bytes_per_value=2)["bytes"]
    6000000
    """
    H_out, W_out, C_out = int(H_out), int(W_out), int(C_out)
    for name, v in (("H_out", H_out), ("W_out", W_out), ("C_out", C_out)):
        if v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v}.")
    bpv = int(bytes_per_value)
    if bpv < 1:
        raise ValueError(f"bytes_per_value must be a positive integer, got {bpv}.")
    bs = int(batch_size)
    if bs < 1:
        raise ValueError(f"batch_size must be a positive integer, got {bs}.")

    dim = H_out * W_out * C_out
    nbytes = dim * bpv

    return RichResult(
        title="Feature-map size",
        summary_lines=[("Shape", (H_out, W_out, C_out)), ("Values", dim),
                       ("MB (batch)", nbytes * bs / 2**20)],
        payload={
            "dim": int(dim),
            "bytes": int(nbytes),
            "megabytes": float(nbytes) / 2**20,
            "batch_bytes": int(nbytes * bs),
            "batch_megabytes": float(nbytes * bs) / 2**20,
            "shape": (H_out, W_out, C_out),
            "estimate": int(dim),
            "n": int(dim),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grfmp: dim = H' * W' * C_out (spatial sizes from grcos); memory scales with the batch"
