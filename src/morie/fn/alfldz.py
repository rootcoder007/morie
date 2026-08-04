# morie.fn -- function file (rootcoder007/morie)
"""Decomposition of the total AlphaFold training loss."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["alphafold_loss_decomposition"]

#: Published coefficients of equation (7).
WEIGHTS = {"fape": 0.5, "aux": 0.5, "dist": 0.3, "msa": 2.0, "conf": 0.01,
           "expres": 0.01, "viol": 1.0}


def alphafold_loss_decomposition(fape, aux, dist, msa, conf, expres=0.0,
                                 viol=0.0, phase="training", ncrop=None):
    """Total per-example loss -- supplement equation (7), p. 32.

    A fixed weighted sum of the FAPE loss, the structure module's auxiliary
    loss, the distogram and masked-MSA cross entropies and the confidence
    loss, with the experimentally-resolved and violation terms added only
    during fine-tuning.  The coefficients are the published ones and are
    not free parameters here.

    Parameters
    ----------
    fape, aux, dist, msa, conf : float
        The five always-present loss terms.
    expres, viol : float
        The two fine-tuning-only terms; ignored when ``phase`` is
        ``"training"``.
    phase : {"training", "finetuning"}
        Which branch of equation (7) to take.
    ncrop : int, optional
        Number of residues after cropping.  When given, the total is
        multiplied by ``sqrt(ncrop)``, the reweighting described just below
        equation (7), which stops short sequences from dominating.

    Returns
    -------
    result : RichResult
        Keys: ``terms`` (each weighted contribution), ``estimate`` (the
        total), ``phase``, ``scale``, ``method``.

    Notes
    -----
    This is an exact closed form, so the parity harness checks it against
    the arithmetic rather than only against the other arm: the total must
    equal the hand-computed weighted sum, and the fine-tuning branch must
    coincide with the training branch when both extra terms are zero.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary equation (7)
    """
    if phase not in ("training", "finetuning"):
        raise ValueError("phase must be 'training' or 'finetuning'")
    terms = {
        "fape": WEIGHTS["fape"] * fape,
        "aux": WEIGHTS["aux"] * aux,
        "dist": WEIGHTS["dist"] * dist,
        "msa": WEIGHTS["msa"] * msa,
        "conf": WEIGHTS["conf"] * conf,
    }
    if phase == "finetuning":
        terms["expres"] = WEIGHTS["expres"] * expres
        terms["viol"] = WEIGHTS["viol"] * viol

    total = sum(terms.values())
    scale = 1.0 if ncrop is None else math.sqrt(float(ncrop))
    return RichResult(
        payload={
            "terms": terms,
            "estimate": total * scale,
            "unscaled": total,
            "phase": phase,
            "scale": scale,
            "method": "AlphaFold total loss decomposition",
        }
    )


def cheatsheet():
    return "alfldz: weighted decomposition of the total AlphaFold loss"
