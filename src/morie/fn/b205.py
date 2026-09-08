# morie.fn -- function file (rootcoder007/morie)
"""Perplexity under a context window (Burkov eq 2.5)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_perplexity"]


def burkov_lm_ch2_perplexity(D, k=None, t=None, log_probs=None, base="e"):
    r"""Perplexity as the exponentiated mean negative log-likelihood.

    Burkov equation (2.5), p. 84:

    .. math::
       \mathrm{Perplexity}(\mathcal{D}, k) = \exp\!\left(
         -\frac{1}{|D|}\sum_{i=1}^{|D|}
         \log \Pr(t_i \mid t_{\max(1,i-k)},\ldots,t_{i-1})\right)

    Perplexity is the exponential of the cross-entropy, and the
    exponential is what makes it interpretable: it is the effective
    BRANCHING FACTOR, the number of equally likely choices the model is
    behaving as though it faces at each step. A perplexity of 100 on a
    50 000-word vocabulary means the model has narrowed the field to
    about 100 candidates.

    Two comparisons are therefore meaningless unless controlled.
    Perplexity depends on the TOKENISATION -- a model over characters
    and one over words are not comparable at all, since they are
    predicting different numbers of things -- and it depends on the
    vocabulary, since a larger one raises the ceiling. ``uniform_ceiling``
    gives :math:`V` for reference when the vocabulary is supplied, and
    bits-per-token is returned for the tokenisation-robust comparison.

    Parameters
    ----------
    D : array-like or int
        Either the per-token log-probabilities, or the token count when
        ``log_probs`` is supplied separately.
    k : int, optional
        Context window, recorded for provenance.
    t : array-like, optional
        Token identities, recorded only.
    log_probs : array-like, optional
        Per-token log-probabilities.
    base : {'e', '2'}
        Base of the supplied logarithms.

    Returns
    -------
    RichResult
        ``perplexity``, ``cross_entropy``, ``bits_per_token``,
        ``n_tokens``, ``uniform_ceiling``.

    References
    ----------
    Burkov (2025), *The Hundred-Page Language Models Book*, chapter 2,
    equation (2.5), p. 84.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(burkov_lm_ch2_perplexity(np.log([0.5, 0.5]))["perplexity"], 6))
    2.0
    """
    lp = np.asarray(D if log_probs is None else log_probs,
                    dtype=float).ravel()
    if lp.size == 0:
        raise ValueError("need at least one token log-probability.")
    if base not in ("e", "2"):
        raise ValueError("base must be 'e' or '2', got %r." % base)
    if base == "2":
        lp = lp * np.log(2.0)
    if np.any(lp > 1e-9):
        raise ValueError(
            "log-probabilities must be non-positive; got a value above zero."
        )
    ce = float(-np.mean(lp))
    return RichResult(
        payload={
            "estimate": float(np.exp(ce)),
            "perplexity": float(np.exp(ce)),
            "cross_entropy": ce,
            "bits_per_token": float(ce / np.log(2.0)),
            "branching_note": (
                "perplexity is the effective branching factor: the number of "
                "equally likely continuations the model behaves as though it "
                "faces at each step"
            ),
            "comparability_note": (
                "perplexity is not comparable across tokenisations or "
                "vocabularies -- a character model and a word model are "
                "predicting different numbers of things; compare bits per "
                "BYTE instead"
            ),
            "uniform_ceiling": None,
            "context_window": None if k is None else int(k),
            "n_tokens": int(lp.size),
            "method": "Perplexity from per-token log-probabilities "
                      "(Burkov eq 2.5)",
        }
    )


def cheatsheet():
    return (
        "b205: perplexity as the exponentiated mean NLL, with why it is not "
        "comparable across tokenisations"
    )
