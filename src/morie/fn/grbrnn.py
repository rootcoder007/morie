# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bidirectional RNN: concatenate forward and backward hidden states."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bidirectional_rnn"]

_METHOD = "Bidirectional RNN state concatenation"


def geron_bidirectional_rnn(h_forward, h_backward, backward_in_reverse_order=False,
                            combine="concat"):
    r"""Combine the two directions of a bidirectional recurrent layer.

    .. math::
        h_t = [\,\overrightarrow{h_t} \;;\; \overleftarrow{h_t}\,]

    The backward layer reads the sequence right-to-left, so
    :math:`\overleftarrow{h_t}` summarises the *future* of position
    ``t``.  A frequent bug is leaving the backward states in the order
    the backward pass emitted them -- last time step first -- which
    silently pairs each position with the wrong half; pass
    ``backward_in_reverse_order=True`` when that is the layout you have.

    Parameters
    ----------
    h_forward : array-like, shape (T, Hf)
        Forward states in time order.
    h_backward : array-like, shape (T, Hb)
        Backward states, in time order unless
        ``backward_in_reverse_order`` says otherwise.
    backward_in_reverse_order : bool, optional
        Flip ``h_backward`` before pairing.
    combine : {"concat", "sum", "mean"}, optional
        How to merge. ``"sum"`` and ``"mean"`` need equal widths.

    Returns
    -------
    RichResult
        Payload keys ``h``, ``output_dim``, ``n_steps``,
        ``forward_dim``, ``backward_dim``, ``estimate`` (mean of the
        combined states), ``n``, ``method``.

    References
    ----------
    Géron Ch 14, Bidirectional RNN section.

    Examples
    --------
    >>> r = geron_bidirectional_rnn([[1.0, 2.0]], [[3.0, 4.0]])
    >>> r["h"]
    [[1.0, 2.0, 3.0, 4.0]]
    >>> r["output_dim"]
    4

    Reversing the backward stack re-pairs the time steps:

    >>> r2 = geron_bidirectional_rnn([[1.0], [2.0]], [[9.0], [8.0]],
    ...                              backward_in_reverse_order=True)
    >>> r2["h"]
    [[1.0, 8.0], [2.0, 9.0]]
    """
    F = np.atleast_2d(np.asarray(h_forward, dtype=float))
    B = np.atleast_2d(np.asarray(h_backward, dtype=float))
    if F.size == 0 or B.size == 0:
        raise ValueError("h_forward and h_backward must be non-empty.")
    if F.shape[0] != B.shape[0]:
        raise ValueError(
            f"h_forward has {F.shape[0]} time steps but h_backward has {B.shape[0]}."
        )
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(B)):
        raise ValueError("hidden states contain non-finite values.")
    if backward_in_reverse_order:
        B = B[::-1]

    if combine == "concat":
        h = np.concatenate([F, B], axis=1)
    elif combine in ("sum", "mean"):
        if F.shape[1] != B.shape[1]:
            raise ValueError(
                f"combine={combine!r} needs equal widths, got {F.shape[1]} and {B.shape[1]}."
            )
        h = F + B
        if combine == "mean":
            h = h / 2.0
    else:
        raise ValueError(
            f"combine must be one of 'concat', 'sum', 'mean', got {combine!r}."
        )

    return RichResult(
        title="Bidirectional RNN states",
        summary_lines=[("Time steps", int(F.shape[0])), ("Output width", int(h.shape[1]))],
        payload={
            "h": h.tolist(),
            "output_dim": int(h.shape[1]),
            "n_steps": int(F.shape[0]),
            "forward_dim": int(F.shape[1]),
            "backward_dim": int(B.shape[1]),
            "combine": combine,
            "estimate": float(h.mean()),
            "n": int(F.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbrnn: bidirectional RNN -- h_t = [h_t_forward ; h_t_backward]"
