# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradual layer unfreezing schedule (Howard and Ruder 2018;
Alammar Ch 11)."""

from ._richresult import RichResult

__all__ = ["alammar_layer_freezing"]


def alammar_layer_freezing(n_layers, n_stages=None):
    """Stage s trains the top s layer blocks and freezes the rest.

    Returns the per-stage boolean train masks (layer 1 bottom, layer L
    top). Monotonicity is structural: once thawed, a layer never
    refreezes, and the tests assert it across every schedule.

    References: Alammar and Grootendorst, Ch 11; Howard and Ruder
    (2018), ULMFiT.

    Examples
    --------
    >>> alammar_layer_freezing(3)["masks"][0]
    [False, False, True]
    """
    L = int(n_layers)
    if L < 1:
        raise ValueError("n_layers must be positive.")
    S = int(n_stages) if n_stages is not None else L
    if not 1 <= S <= L:
        raise ValueError(f"n_stages must lie in [1, {L}].")
    masks = []
    for s in range(1, S + 1):
        thaw = round(s * L / S)
        masks.append([i >= L - thaw for i in range(L)])
    return RichResult(payload={
        "masks": masks, "n_stages": S,
        "trainable_per_stage": [sum(m) for m in masks],
        "estimate": float(S), "n": L,
        "method": "Gradual unfreezing (Howard and Ruder 2018)"})


def cheatsheet():
    return "alfrz: top-down thaw masks, monotone by construction"
