# morie.fn -- function file (rootcoder007/morie)
"""Noise-contrastive estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_nce_loss", "burkov_noise_contrastive_estimation"]


def burkov_nce_loss(pos_scores, neg_scores, noise_prob=None, k=None):
    r"""Binary discrimination of data against noise.

    .. math::
       J = -\mathbb{E}\Big[\log\sigma(s^{+})
           + \sum_{i=1}^{k}\log\big(1-\sigma(s^{-}_i)\big)\Big]

    NCE replaces the softmax over a vocabulary of size :math:`V` with
    :math:`k+1` binary decisions, turning an :math:`O(V)` normalisation
    into an :math:`O(k)` one. That is the entire reason it exists:
    with :math:`V` in the hundreds of thousands the partition function
    is the dominant cost of training.

    The trick is that the model no longer needs a normalised
    distribution. Under the proper NCE objective the score includes a
    correction :math:`\log(k\,q(w))` for the noise distribution --
    note the POSITIVE term approximates :math:`q(w)` of the true word
    by the mean noise probability of that row's sampled negatives,
    since the true word's own noise probability is not an input -- and
    with it the learned scores converge to the true log-probabilities
    up to a constant. Dropping that correction gives NEGATIVE SAMPLING,
    which is what word2vec actually uses -- cheaper, and no longer a
    consistent estimator of the language model, which is fine when only
    the embeddings are wanted and not fine when the probabilities are.
    ``corrected`` records which of the two was computed.

    Parameters
    ----------
    pos_scores : array-like, shape (n,)
        Model score for the true word.
    neg_scores : array-like, shape (n, k)
        Scores for the sampled noise words.
    noise_prob : array-like, shape (n, k), optional
        Noise distribution probabilities. Supplying them applies the
        NCE correction; omitting them gives negative sampling.
    k : int, optional
        Checked against ``neg_scores``.

    Returns
    -------
    RichResult
        ``loss``, ``pos_loss``, ``neg_loss``, ``corrected``,
        ``accuracy``, ``cost_ratio``.

    References
    ----------
    Burkov (2025), *The Language Model Book*, chapter 2,
    noise-contrastive estimation.
    Gutmann and Hyvarinen (2010), *AISTATS*. Mnih and Teh (2012).
    Mikolov et al. (2013) for negative sampling.

    Examples
    --------
    >>> out = burkov_nce_loss([10.0], [[-10.0]])
    >>> bool(out["loss"] < 1e-4)
    True
    """
    pos = np.asarray(pos_scores, dtype=float).ravel()
    neg = np.atleast_2d(np.asarray(neg_scores, dtype=float))
    n = pos.size
    if neg.shape[0] != n:
        raise ValueError(
            "neg_scores has %d rows for %d positives." % (neg.shape[0], n)
        )
    kk = neg.shape[1]
    if k is not None and int(k) != kk:
        raise ValueError(
            "k says %d but neg_scores has %d columns." % (int(k), kk)
        )
    corrected = noise_prob is not None
    if corrected:
        Q = np.atleast_2d(np.asarray(noise_prob, dtype=float))
        if Q.shape != neg.shape:
            raise ValueError(
                "noise_prob must match neg_scores in shape, got %s and %s."
                % (Q.shape, neg.shape)
            )
        if np.any(Q <= 0):
            raise ValueError("noise probabilities must be positive.")
        neg_adj = neg - np.log(kk * Q)
        pos_adj = pos - np.log(kk * np.maximum(Q.mean(axis=1), 1e-300))
    else:
        neg_adj, pos_adj = neg, pos

    def logsig(z):
        return -np.logaddexp(0.0, -z)

    pl = float(-np.mean(logsig(pos_adj)))
    nl = float(-np.mean(np.sum(logsig(-neg_adj), axis=1)))
    acc = float(np.mean(pos_adj[:, None] > neg_adj))
    return RichResult(
        payload={
            "estimate": pl + nl,
            "loss": pl + nl,
            "pos_loss": pl,
            "neg_loss": nl,
            "accuracy": acc,
            "corrected": corrected,
            "objective": ("noise-contrastive estimation" if corrected
                          else "negative sampling"),
            "correction_note": (
                "with the log(k q(w)) correction the learned scores converge "
                "to true log-probabilities up to a constant; without it this "
                "is negative sampling, which is cheaper but no longer a "
                "consistent estimator of the language model"
                if not corrected else
                "the log(k q(w)) correction is applied, so the scores "
                "estimate log-probabilities up to a constant"
            ),
            "k": int(kk),
            "cost_ratio": float(kk + 1),
            "cost_note": (
                "%d binary decisions replace a softmax over the whole "
                "vocabulary" % (kk + 1)
            ),
            "n": int(n),
            "method": "Noise-contrastive estimation loss",
        }
    )


def cheatsheet():
    return (
        "bknce: NCE loss, and the log(kq) correction that separates it from "
        "negative sampling"
    )


#: Catalogue alias for :func:`burkov_nce_loss`.
burkov_noise_contrastive_estimation = burkov_nce_loss
