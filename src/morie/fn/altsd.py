# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TSDAE denoising objective bookkeeping (Wang et al. 2021;
Alammar Ch 10)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_tsdae_objective"]


def alammar_tsdae_objective(tokens, delete_ratio=0.6, seed=1,
                            reconstruction_logprob=None):
    """Corrupt by token deletion at the stated ratio (the paper's
    optimal 0.6), on the shared LCG so both languages delete the same
    tokens; the loss is -sum log p(x_t | Enc(corrupt(x)), x_<t),
    evaluated when the caller supplies per-token log-probs.

    References: Alammar and Grootendorst, Ch 10; Wang, Reimers and
    Gurevych (2021).
    """
    toks = [str(t) for t in tokens]
    if not toks:
        raise ValueError("no tokens supplied.")
    r = float(delete_ratio)
    if not 0 < r < 1:
        raise ValueError("delete_ratio must lie in (0, 1).")
    s = int(seed) % 2 ** 32
    kept = []
    deleted = []
    for t in toks:
        s = (1664525 * s + 1013904223) % 2 ** 32
        u = (s + 0.5) / 2 ** 32
        (deleted if u < r else kept).append(t)
    if not kept:
        kept = [toks[0]]      # never hand the encoder an empty input
        deleted = toks[1:]
    loss = None
    if reconstruction_logprob is not None:
        lps = [float(v) for v in reconstruction_logprob]
        if len(lps) != len(toks):
            raise ValueError(
                "need one reconstruction log-prob per ORIGINAL token; "
                "the decoder must rebuild the uncorrupted sentence.")
        loss = -sum(lps)
    return RichResult(payload={
        "corrupted": kept, "deleted": deleted,
        "actual_delete_ratio": len(deleted) / len(toks),
        "loss": loss,
        "estimate": loss if loss is not None else float(len(deleted)),
        "n": len(toks),
        "method": "TSDAE deletion corruption + NLL (Wang et al. 2021)"})


def cheatsheet():
    return "altsd: LCG-driven deletion at 0.6, NLL over the ORIGINAL tokens"
