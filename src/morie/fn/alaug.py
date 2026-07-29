# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Augmented SBERT: cross-encoder silver labels train the
bi-encoder (Thakur et al. 2021; Alammar Ch 10)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_augmented_sbert"]


def alammar_augmented_sbert(unlabeled_pairs, cross_encoder,
                            gold_pairs=None, gold_labels=None):
    """silver = CrossEncoder(pairs); the bi-encoder trains on gold +
    silver. The labelling step is computed here; the payload separates
    gold from silver counts and, when gold is supplied, reports the
    cross-encoder's agreement with it -- silver from a judge that
    cannot reproduce gold is noise amplification, not augmentation.

    References: Alammar and Grootendorst, Ch 10; Thakur et al. (2021).
    """
    if not callable(cross_encoder):
        raise ValueError("cross_encoder must be callable "
                         "(text_a, text_b) -> score.")
    pairs = [(str(a), str(b)) for a, b in unlabeled_pairs]
    if not pairs:
        raise ValueError("no unlabeled pairs supplied.")
    silver = [float(cross_encoder(a, b)) for a, b in pairs]
    agreement = None
    if gold_pairs is not None and gold_labels is not None:
        gp = [(str(a), str(b)) for a, b in gold_pairs]
        gl = [float(v) for v in gold_labels]
        if len(gp) != len(gl):
            raise ValueError("gold pairs and labels must align.")
        pred = np.array([float(cross_encoder(a, b)) for a, b in gp])
        gold = np.array(gl)
        if len(gp) >= 2 and gold.std() > 0 and pred.std() > 0:
            agreement = float(np.corrcoef(pred, gold)[0, 1])
    training_set = [{"pair": p, "label": s, "source": "silver"}
                    for p, s in zip(pairs, silver)]
    if gold_pairs is not None and gold_labels is not None:
        training_set += [{"pair": p, "label": float(l), "source": "gold"}
                         for p, l in zip(gp, gl)]
    return RichResult(payload={
        "training_set": training_set,
        "n_silver": len(pairs),
        "n_gold": 0 if gold_pairs is None else len(gp),
        "cross_encoder_gold_agreement": agreement,
        "estimate": float(np.mean(silver)), "n": len(training_set),
        "method": "Augmented SBERT silver labelling (Thakur et al. 2021)"})


def cheatsheet():
    return "alaug: silver labels from the cross-encoder, gold agreement measured"
