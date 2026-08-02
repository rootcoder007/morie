# morie.fn -- function file (rootcoder007/morie)
"""Dataset-level binary cross-entropy (Burkov eq 1.10)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_dataset_bce"]


def burkov_lm_ch1_dataset_bce(y_hat, y, N=None, eps=1e-12):
    r"""Average binary cross-entropy over a dataset.

    Burkov equation (1.10), p. 41:

    .. math::
       \mathrm{loss}_{\mathcal{D}} = -\frac{1}{N}\sum_{i=1}^{N}
         \big[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\big]

    The division by :math:`N` is what makes the loss comparable across
    datasets of different size and keeps the gradient scale independent
    of batch size -- a summed loss would make the learning rate depend
    on how many examples happened to be in the batch.

    Predictions are clipped away from 0 and 1 before taking logs. A
    confident wrong prediction otherwise contributes an infinite loss
    and destroys the average, which is a numerical artefact rather than
    a fact about the model. ``n_clipped`` reports how often that
    mattered; a large count means the model is saturating and the
    gradient has effectively vanished there.

    ``baseline_loss`` is the entropy of the label distribution -- what a
    model predicting the base rate would score. A loss above it means
    the model is worse than a constant.

    Parameters
    ----------
    y_hat : array-like, shape (N,)
        Predicted probabilities of the positive class.
    y : array-like of {0, 1}, shape (N,)
    N : int, optional
        Checked against the data length.
    eps : float
        Clipping bound.

    Returns
    -------
    RichResult
        ``loss``, ``per_example``, ``baseline_loss``, ``skill``,
        ``n_clipped``, ``accuracy``.

    References
    ----------
    Burkov (2025), *The Hundred-Page Language Models Book*, chapter 1,
    equation (1.10), p. 41.

    Examples
    --------
    >>> round(float(burkov_lm_ch1_dataset_bce([0.5, 0.5], [1, 0])["loss"]), 6)
    0.693147
    """
    p = np.asarray(y_hat, dtype=float).ravel()
    t = np.asarray(y, dtype=float).ravel()
    if p.size != t.size:
        raise ValueError(
            "y_hat and y must agree in length, got %d and %d."
            % (p.size, t.size)
        )
    n = p.size
    if n == 0:
        raise ValueError("need at least one example.")
    if N is not None and int(N) != n:
        raise ValueError("N says %d but the data has %d rows." % (int(N), n))
    if not np.all(np.isin(t, (0.0, 1.0))):
        raise ValueError("y must be binary 0/1.")
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9):
        raise ValueError("y_hat must be probabilities in [0, 1].")
    clipped = int(np.sum((p < eps) | (p > 1 - eps)))
    pc = np.clip(p, eps, 1 - eps)
    per = -(t * np.log(pc) + (1 - t) * np.log(1 - pc))
    loss = float(np.mean(per))
    base = float(np.mean(t))
    bc = np.clip(base, eps, 1 - eps)
    baseline = float(-(base * np.log(bc) + (1 - base) * np.log(1 - bc)))
    return RichResult(
        payload={
            "estimate": loss,
            "loss": loss,
            "per_example": per,
            "baseline_loss": baseline,
            "skill": float(1.0 - loss / baseline) if baseline > 0 else np.nan,
            "skill_note": (
                "1 - loss/baseline; at or below zero the model is no better "
                "than predicting the base rate"
            ),
            "n_clipped": clipped,
            "clip_note": (
                "a confident wrong prediction would otherwise contribute an "
                "infinite loss; many clipped values mean the model is "
                "saturating and its gradient has vanished there"
            ),
            "accuracy": float(np.mean((p >= 0.5) == (t == 1))),
            "base_rate": base,
            "N": int(n),
            "method": "Dataset-average binary cross-entropy (Burkov eq 1.10)",
        }
    )


def cheatsheet():
    return "b110: mean binary cross-entropy with a base-rate skill score"
