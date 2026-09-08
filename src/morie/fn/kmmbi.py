# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Membership inference: threshold the model's loss to guess
training-set membership."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_membership_inference"]


def kamath_membership_inference(losses, threshold, labels=None):
    """member_hat(x) = 1 if L_model(x) < tau else 0.

    Strictly below tau, so a loss exactly at the threshold is called a
    non-member -- an arbitrary choice, which is why it is stated here
    instead of left in the code. Pass ``labels`` (1 = truly a member)
    to get the attack's accuracy, TPR and FPR; without them only the
    predictions are returned, never an invented success rate.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 6,
    membership inference; that section is not in the 2024 PDF, so the
    rule is implemented exactly as the spec line states (Shokri et al.
    2017; Carlini et al. 2021).

    Examples
    --------
    >>> out = kamath_membership_inference([0.1, 0.5, 2.0], 1.0,
    ...                                   labels=[1, 1, 0])
    >>> out["predictions"]
    [1, 1, 0]
    >>> out["estimate"]
    1.0
    >>> out["tpr"], out["fpr"]
    (1.0, 0.0)
    """
    L = np.atleast_1d(np.asarray(losses, dtype=float)).ravel()
    tau = float(threshold)
    if L.size == 0:
        raise ValueError("no losses supplied.")
    if not np.all(np.isfinite(L)):
        raise ValueError(
            "a non-finite loss cannot be compared with a threshold.")
    pred = (L < tau).astype(int)
    payload = {
        "predictions": [int(v) for v in pred],
        "n_predicted_members": int(pred.sum()),
        "member_rate": float(pred.mean()),
        "threshold": tau, "n": int(L.size),
        "estimate": float(pred.mean()),
        "method": "Membership inference by loss threshold"}
    if labels is not None:
        y = np.atleast_1d(np.asarray(labels)).ravel().astype(int)
        if y.size != L.size:
            raise ValueError(
                f"got {y.size} labels for {L.size} losses.")
        if not np.all(np.isin(y, (0, 1))):
            raise ValueError("labels must be 0 (non-member) or 1 (member).")
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        if pos == 0 or neg == 0:
            raise ValueError(
                "the labels contain only one class, so TPR or FPR is "
                "0/0; an attack cannot be scored on it.")
        payload["accuracy"] = float(np.mean(pred == y))
        payload["tpr"] = tp / pos
        payload["fpr"] = fp / neg
        payload["advantage"] = payload["tpr"] - payload["fpr"]
        payload["estimate"] = payload["accuracy"]
    return RichResult(payload=payload)


def cheatsheet():
    return "kmmbi: predict member iff loss < tau; TPR/FPR only with labels"
