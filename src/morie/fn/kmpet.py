# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PET (Pattern-Exploiting Training): verbalizer cross-entropy plus an
auxiliary MLM term."""

from . import _array_core as np

from ._richresult import RichResult
from .km022 import kamath_ch2_mlm_loss

__all__ = ["kamath_pet_loss"]

_IGNORE = -100


def _log_softmax(z):
    m = z.max()
    return z - (m + np.log(np.exp(z - m).sum()))


def kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets,
                    alpha, ignore_index=_IGNORE):
    """L_PET = L_CE(verbalizer, y_true) + alpha * L_MLM(masked tokens).

    The auxiliary term is the ordinary masked-LM loss, so it is
    DELEGATED to ``morie.fn.km022`` (Kamath Eq 2.22) after turning the
    logits into the probability of the true token at each masked
    position. ``mlm_targets`` marks unmasked positions with
    ``ignore_index``; if nothing is masked, alpha multiplies a loss
    that does not exist, so that is refused rather than treated as 0.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, PET / iPET
    (Schick and Schutze 2021).

    Examples
    --------
    >>> import math
    >>> out = kamath_pet_loss([0.0, 0.0], 0, [[0.0, 0.0]], [1], 1.0)
    >>> abs(out["estimate"] - 2 * math.log(2)) < 1e-12
    True
    >>> abs(out["loss_ce"] - math.log(2)) < 1e-12
    True
    >>> out2 = kamath_pet_loss([0.0, 0.0], 0, [[0.0, 0.0]], [1], 0.0)
    >>> abs(out2["estimate"] - math.log(2)) < 1e-12
    True
    """
    vz = np.atleast_1d(np.asarray(verbalizer_logits, dtype=float)).ravel()
    alpha = float(alpha)
    if vz.size < 2:
        raise ValueError(
            "the verbalizer must score at least two classes.")
    if not np.all(np.isfinite(vz)):
        raise ValueError("a verbalizer logit is non-finite.")
    y = int(y_true)
    if not 0 <= y < vz.size:
        raise ValueError(f"y_true must lie in [0, {vz.size - 1}]; got {y}.")
    if alpha < 0:
        raise ValueError(
            f"alpha must be non-negative; got {alpha}. A negative "
            "weight rewards the model for being wrong on the masked "
            "tokens.")
    ce = float(-_log_softmax(vz)[y])

    ml = np.atleast_2d(np.asarray(mlm_logits, dtype=float))
    mt = np.atleast_1d(np.asarray(mlm_targets)).ravel().astype(int)
    if mt.size != ml.shape[0]:
        raise ValueError(
            f"mlm_targets has {mt.size} entries for {ml.shape[0]} "
            "positions.")
    masked = np.flatnonzero(mt != ignore_index)
    if masked.size == 0:
        raise ValueError(
            "no position is masked, so there is no MLM term to weight; "
            "pass alpha = 0 and a real mask, or use the plain CE loss.")
    if np.any((mt[masked] < 0) | (mt[masked] >= ml.shape[1])):
        raise ValueError(
            f"a masked target id lies outside [0, {ml.shape[1] - 1}].")
    p_true = np.ones(ml.shape[0])
    for t in masked:
        p_true[t] = float(np.exp(_log_softmax(ml[t])[mt[t]]))
    mlm = kamath_ch2_mlm_loss(p_true, masked)
    total = ce + alpha * float(mlm["estimate"])
    return RichResult(payload={
        "estimate": total, "loss": total,
        "loss_ce": ce, "loss_mlm": float(mlm["estimate"]),
        "alpha": alpha, "n_masked": int(masked.size),
        "n": int(ml.shape[0]),
        "method": "PET loss = verbalizer CE + alpha * MLM (via km022)"})


def cheatsheet():
    return "kmpet: CE(verbalizer) + alpha * km022's MLM loss"


# compact alias per ledger/NAMING.md
kamathpetloss = kamath_pet_loss
