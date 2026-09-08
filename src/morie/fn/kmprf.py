# morie.fn -- function file (rootcoder007/morie)
"""Prefix-LM attention mask: bidirectional over the prefix, causal over the completion."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_prefix_lm_mask"]


def kamath_prefix_lm_mask(prefix_len, total_len):
    r"""Boolean attention mask for a prefix language model.

    Position i may attend to position j when

    .. math:: j < P \quad\text{(the prefix is fully visible)}
              \qquad\text{or}\qquad j \le i,

    so tokens inside the prefix see each other in both directions
    while completion tokens keep the strict causal restriction. With
    P = 0 this reduces to the standard lower-triangular causal mask;
    with P = L it is fully bidirectional.

    Parameters
    ----------
    prefix_len : int
        Number of prefix positions P, 0 <= P <= L.
    total_len : int
        Sequence length L.

    Returns
    -------
    RichResult
        keys: ``mask`` (L, L) boolean, True = attention allowed,
        ``additive`` (0 where allowed, -inf where blocked, for adding
        to logits), ``prefix_len``, ``total_len``, ``n_allowed``,
        ``method``.

    References
    ----------
    Kamath, U., Graham, K. L. & Emara, W. (2022). *Transformers for
    Machine Learning: A Deep Dive*. Chapman & Hall/CRC. Ch. 3
    (attention masking; prefix-LM vs causal-LM objectives).
    """
    P = int(prefix_len)
    L = int(total_len)
    if L < 1:
        raise ValueError(f"total_len must be positive, got {L}.")
    if not 0 <= P <= L:
        raise ValueError(f"prefix_len must lie in [0, {L}], got {P}.")

    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    mask = (j < P) | (j <= i)
    additive = np.where(mask, 0.0, -np.inf)

    return RichResult(
        payload={
            "mask": mask,
            "additive": additive,
            "prefix_len": P,
            "total_len": L,
            "n_allowed": int(mask.sum()),
            "method": "Prefix-LM mask: bidirectional within the prefix, causal after it",
        }
    )


def cheatsheet():
    return "kmprf: allow j < P or j <= i; P=0 gives the causal mask, P=L bidirectional"
