# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stride: step size of kernel sliding over input."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_stride"]


def geron_stride(in_dim, k, p=0, s=1):
    """
    Stride: step size of kernel sliding over input.

    Formula: output_dim = floor((in_dim + 2p - k)/s) + 1

    Also reports the receptive-field arithmetic that goes with it: the
    "same"-padding that would keep the size at ``ceil(in_dim/s)``, and
    how many input positions fall off the end because the window does
    not divide evenly (dropped = in_dim + 2p - k - s*(out-1) - 1).

    Parameters
    ----------
    in_dim : int
        Input length along the axis (>= 1).
    k : int
        Kernel size (1 <= k <= in_dim + 2p).
    p : int, default 0
        Zero padding on each side (>= 0).
    s : int, default 1
        Stride (>= 1).

    Returns
    -------
    result : RichResult
        Keys: output_dim, dropped, same_padding, estimate, n, method.

    Examples
    --------
    AlexNet's first layer: 227 input, 11 kernel, stride 4, no padding.

    >>> int(geron_stride(227, 11, 0, 4)["output_dim"])
    55
    >>> int(geron_stride(28, 3, 1, 1)["output_dim"])
    28
    >>> r = geron_stride(28, 3, 0, 2)
    >>> int(r["output_dim"]), int(r["dropped"])
    (13, 1)

    References
    ----------
    Géron Ch 12
    """
    n_in, kk, pp, ss = int(in_dim), int(k), int(p), int(s)
    if n_in < 1:
        raise ValueError(f"geron_stride: in_dim must be >= 1, got {n_in}")
    if kk < 1:
        raise ValueError(f"geron_stride: kernel size k must be >= 1, got {kk}")
    if pp < 0:
        raise ValueError(f"geron_stride: padding p must be >= 0, got {pp}")
    if ss < 1:
        raise ValueError(f"geron_stride: stride s must be >= 1, got {ss}")
    padded = n_in + 2 * pp
    if kk > padded:
        raise ValueError(
            f"geron_stride: kernel {kk} does not fit in the padded input {padded} "
            f"(in_dim={n_in}, p={pp}); the output would be empty"
        )

    out = (padded - kk) // ss + 1
    dropped = padded - kk - ss * (out - 1)
    same_total = max(0, ss * (-(-n_in // ss) - 1) + kk - n_in)

    return RichResult(
        title="Stride arithmetic",
        summary_lines=[("Output dim", out), ("Dropped positions", dropped), ("Padding for 'same'", same_total)],
        interpretation=(
            "Stride s downsamples by roughly s; positions are dropped whenever the window does not "
            "tile the padded input exactly."
        ),
        payload={
            "output_dim": int(out),
            "dropped": int(dropped),
            "same_padding": int(same_total),
            "padded_dim": int(padded),
            "in_dim": n_in,
            "k": kk,
            "p": pp,
            "s": ss,
            "estimate": float(out),
            "n": int(n_in),
            "method": "Convolution output size floor((in + 2p - k)/s) + 1",
        },
    )


def cheatsheet():
    return "hmstr2: Stride: step size of kernel sliding over input"


# compact alias per ledger/NAMING.md
geronstride = geron_stride
