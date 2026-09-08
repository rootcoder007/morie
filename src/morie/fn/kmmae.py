# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multimodal masked autoencoder: squared reconstruction error on the
masked patches."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_multimodal_mae"]


def _as_modality_dict(obj, name):
    if isinstance(obj, dict):
        if not obj:
            raise ValueError(f"{name} is an empty modality dict.")
        return dict(obj)
    return {"default": obj}


def kamath_multimodal_mae(x_visible, x_masked_true, masks, decoders=None):
    """L = sum_m || x_m - Decoder_m(Encoder([x_visible, mask_tokens])) ||^2.

    Each argument is a dict keyed by modality (a bare array is treated
    as the single modality "default"). ``decoders`` maps the same keys
    to callables ``(visible, mask) -> reconstruction`` shaped like the
    masked ground truth; there is no default decoder, because a
    reconstruction loss without a reconstruction is not a loss.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 9,
    multimodal masked autoencoder; that section is not in the 2024
    PDF, so the objective is implemented exactly as the spec line
    states (He et al. 2022 MAE, extended per modality).

    Examples
    --------
    >>> vis = {"img": [1.0, 1.0]}
    >>> true = {"img": [2.0, 4.0]}
    >>> msk = {"img": [False, False, True, True]}
    >>> dec = {"img": lambda v, m: [2.0, 3.0]}
    >>> out = kamath_multimodal_mae(vis, true, msk, decoders=dec)
    >>> out["estimate"]
    1.0
    >>> out["n_masked"]
    2
    """
    if decoders is None:
        raise ValueError(
            "decoders is required: pass {modality: callable(visible, "
            "mask) -> reconstruction}.")
    vis = _as_modality_dict(x_visible, "x_visible")
    true = _as_modality_dict(x_masked_true, "x_masked_true")
    msk = _as_modality_dict(masks, "masks")
    dec = _as_modality_dict(decoders, "decoders")
    keys = list(vis.keys())
    for other, nm in ((true, "x_masked_true"), (msk, "masks"),
                      (dec, "decoders")):
        if set(other.keys()) != set(keys):
            raise ValueError(
                f"{nm} covers modalities {sorted(other.keys())} but "
                f"x_visible covers {sorted(keys)}.")

    per_modality = {}
    total = 0.0
    n_masked = 0
    for m in keys:
        f = dec[m]
        if not callable(f):
            raise ValueError(f"the decoder for modality {m!r} is not callable.")
        mask = np.asarray(msk[m])
        if mask.dtype != bool and not np.all(np.isin(mask, (0, 1))):
            raise ValueError(
                f"the mask for modality {m!r} must be boolean or 0/1.")
        mask = mask.astype(bool)
        if not mask.any():
            raise ValueError(
                f"modality {m!r} has nothing masked; an autoencoder "
                "that reconstructs only what it was shown learns "
                "nothing.")
        gold = np.atleast_1d(np.asarray(true[m], dtype=float)).ravel()
        if gold.size != int(mask.sum()):
            raise ValueError(
                f"modality {m!r}: {int(mask.sum())} positions are masked "
                f"but {gold.size} ground-truth values were supplied.")
        rec = np.atleast_1d(np.asarray(
            f(vis[m], mask), dtype=float)).ravel()
        if rec.size != gold.size:
            raise ValueError(
                f"the decoder for {m!r} returned {rec.size} values for "
                f"{gold.size} masked positions.")
        sse = float(np.sum((gold - rec) ** 2))
        per_modality[m] = sse
        total += sse
        n_masked += gold.size
    return RichResult(payload={
        "estimate": total, "loss": total,
        "per_modality": per_modality,
        "modalities": keys, "n_masked": int(n_masked),
        "n": len(keys),
        "method": "Multimodal MAE squared reconstruction loss"})


def cheatsheet():
    return "kmmae: sum_m ||x_masked - Decoder_m(visible)||^2, decoders required"
