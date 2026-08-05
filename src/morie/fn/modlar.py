# morie.fn -- function file (rootcoder007/morie)
"""Modularity Q of a community partition (alias of :mod:`sgtmodq`)."""

from .sgtmodq import sgt_modularity_q

__all__ = ["modularity_newman", "modularitynewman"]


def modularity_newman(A, communities):
    """Modularity ``Q`` of a community partition (Newman's form).

    This module is an ALIAS.  Modularity is implemented once, in
    ``sgtmodq.sgt_modularity_q``; this entry point delegates.  It is the
    same quantity as :func:`morie.fn.modulq.modularity_q`; the two module
    names differ only in which of the two papers they cite.

        Q = (1 / 2m) sum_ij [A_ij - k_i k_j / 2m] delta(c_i, c_j)

    The stub this replaces took a leading ``y`` data argument that its
    body only averaged; the argument carried no meaning for modularity
    and has been dropped.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric adjacency or weight matrix.
    communities : array-like of int, length n
        Community label per node.

    Returns
    -------
    RichResult
        ``Q``, ``estimate``, ``n_communities``, ``n``.

    References
    ----------
    Newman, M. E. J. (2006), "Modularity and community structure in
    networks", PNAS 103(23), 8577-8582, doi:10.1073/pnas.0601602103.
    """
    return sgt_modularity_q(A, communities)


modularitynewman = modularity_newman


def cheatsheet():
    return "modlar: Modularity Q (Newman) of a community partition (alias of sgtmodq)"
