# morie.fn -- function file (rootcoder007/morie)
"""Nucleus (top-p) sampling.

DUPLICATE.  Nucleus sampling -- Holtzman, A., Buys, J., Du, L., Forbes,
M. & Choi, Y. (2020), "The curious case of neural text degeneration",
ICLR 2020 -- is already implemented in ``morie.fn.toppd`` as
``top_p_nucleus``: temperature softmax, descending sort, smallest prefix
whose cumulative probability reaches ``p``, renormalise.  Confirmed the
same method by reading both docstrings and the truncation rule, as
``ledger/wave2/SKIP_README.md`` requires before aliasing.

This module is kept as a name alias so callers who reach for ``toppS``
land on the one implementation rather than a second copy.
"""

from .toppd import top_p_nucleus as _impl

__all__ = ["top_p_sampling"]


def top_p_sampling(logits, p, temp):
    """Nucleus (top-p) truncated softmax.

    Alias of :func:`morie.fn.toppd.top_p_nucleus`.

    Parameters
    ----------
    logits : array-like, shape (V,)
        Unnormalised scores.
    p : float
        Cumulative-probability cutoff in ``(0, 1]``.
    temp : float
        Softmax temperature.

    Returns
    -------
    RichResult
        Whatever :func:`morie.fn.toppd.top_p_nucleus` returns:
        ``tensor``, ``keep_mask``, ``n_kept``, ``p``, ``method``.

    References
    ----------
    Holtzman, A., Buys, J., Du, L., Forbes, M. & Choi, Y. (2020).  The
    curious case of neural text degeneration.  International Conference
    on Learning Representations.  arXiv:1904.09751.
    """
    return _impl(logits, p, temp)


def cheatsheet():
    return "toppS: alias of toppd.top_p_nucleus (nucleus sampling)."

# public names resolved by fn/_lazy_map.json
toppsampling = top_p_sampling
