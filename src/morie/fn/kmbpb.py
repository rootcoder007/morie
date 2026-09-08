# morie.fn -- function file (rootcoder007/morie)
"""Bits-per-byte."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bits_per_byte", "kamath_bits_per_byte"]


def bits_per_byte(log_probs, n_bytes=None, bytes_per_token=None, base="e"):
    r"""Cross-entropy normalised by BYTES rather than tokens.

    .. math::
       \mathrm{BPB} = \frac{\mathrm{CE}_{\text{nats}}}
                           {\ln 2 \cdot \bar b}

    with :math:`\bar b` the average bytes per token.

    This exists because perplexity is not comparable across
    tokenisers. A model with a large vocabulary packs more text into
    each token, so it predicts fewer, easier-to-predict units and posts
    a lower perplexity WITHOUT being a better model of the text. Bytes
    are a property of the text rather than of the tokeniser, so
    normalising by them removes the incentive to win by tokenisation.

    ``perplexity`` is returned alongside precisely so the gap is
    visible: two models with very different perplexities can have
    nearly identical bits-per-byte, and when they do, the perplexity
    difference was about vocabulary, not modelling.

    Parameters
    ----------
    log_probs : array-like, shape (T,)
        Per-token log-probabilities of the observed tokens.
    n_bytes : int, optional
        Total bytes of the decoded text. Supply this or
        ``bytes_per_token``.
    bytes_per_token : float or array-like, optional
    base : {'e', '2'}

    Returns
    -------
    RichResult
        ``bits_per_byte``, ``bits_per_token``, ``perplexity``,
        ``cross_entropy``, ``bytes_per_token``, ``compression_ratio``.

    References
    ----------
    Kamath, Keenan, Somers and Sorenson (2024), *Large Language
    Models: A Deep Dive*, Springer, chapter 8, bits-per-byte.
    Gao et al. (2020), "The Pile", arXiv:2101.00027, for the standard
    definition.

    Examples
    --------
    >>> import numpy as np
    >>> out = bits_per_byte(np.log([0.5, 0.5]), bytes_per_token=1.0)
    >>> float(out["bits_per_byte"])
    1.0
    """
    lp = np.asarray(log_probs, dtype=float).ravel()
    if lp.size == 0:
        raise ValueError("need at least one token log-probability.")
    if base not in ("e", "2"):
        raise ValueError("base must be 'e' or '2', got %r." % base)
    if base == "2":
        lp = lp * np.log(2.0)
    if np.any(lp > 1e-9):
        raise ValueError("log-probabilities must be non-positive.")
    T = lp.size
    if n_bytes is None and bytes_per_token is None:
        raise ValueError(
            "supply n_bytes or bytes_per_token: bits-per-byte cannot be "
            "formed without knowing how much text the tokens covered."
        )
    if n_bytes is not None:
        nb = float(n_bytes)
        if nb <= 0:
            raise ValueError("n_bytes must be positive.")
        bpt = nb / T
    else:
        b = np.asarray(bytes_per_token, dtype=float).ravel()
        if b.size not in (1, T):
            raise ValueError(
                "bytes_per_token must be a scalar or one value per token."
            )
        if np.any(b <= 0):
            raise ValueError("bytes_per_token must be positive.")
        bpt = float(np.mean(b))
        nb = bpt * T

    ce = float(-np.sum(lp))            # total nats
    bits = ce / np.log(2.0)
    return RichResult(
        payload={
            "estimate": float(bits / nb),
            "bits_per_byte": float(bits / nb),
            "bits_per_token": float(bits / T),
            "cross_entropy": float(ce / T),
            "perplexity": float(np.exp(ce / T)),
            "bytes_per_token": float(bpt),
            "compression_ratio": float(8.0 * nb / bits) if bits > 0 else np.inf,
            "compression_note": (
                "how many times smaller than raw 8-bit bytes an arithmetic "
                "coder using this model would make the text"
            ),
            "comparability_note": (
                "perplexity rewards a larger vocabulary, which packs more "
                "text per token and so predicts fewer, easier units; bytes "
                "belong to the text rather than the tokeniser, which is why "
                "this is the cross-model comparison"
            ),
            "n_tokens": int(T),
            "n_bytes": float(nb),
            "method": "Bits-per-byte",
        }
    )


def cheatsheet():
    return (
        "kmbpb: bits-per-byte, the tokeniser-independent counterpart of "
        "perplexity"
    )


#: Catalogue alias for :func:`bits_per_byte`.
kamath_bits_per_byte = bits_per_byte
