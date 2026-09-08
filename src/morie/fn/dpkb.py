# morie.fn -- function file (rootcoder007/morie)
"""k-anonymity baseline (NOT differential privacy).

DUPLICATE.  The method this module names -- Sweeney, L. (2002),
"k-anonymity: a model for protecting privacy", *International Journal of
Uncertainty, Fuzziness and Knowledge-Based Systems* 10(5), 557-570,
doi:10.1142/S0218488502001648 -- is already implemented in
``morie.fn.kanon`` as ``k_anonymity_check``, together with its R arm
``Kanon``.  A release satisfies k-anonymity when every combination of
quasi-identifier values that appears in it appears at least k times, so
there is exactly one thing to compute and it is computed there.

This module therefore ALIASES the shipped implementation rather than
writing a second copy of it: a second copy would agree with the first at
1e-9 forever and be indistinguishable from correct work while doubling
the surface.  The public name and the payload keys are unchanged.

The name is also a warning: k-anonymity is a syntactic property of a
released table, not a differential privacy guarantee.  It says nothing
about what an adversary with side information can learn, which is what
the ``dp*`` mechanisms in this package bound.
"""

from __future__ import annotations

from .kanon import k_anonymity_check as _impl

__all__ = ["k_anonymity"]


def k_anonymity(X, quasi_ids, k):
    """Smallest equivalence class size on the quasi-identifiers, and k.

    Alias of :func:`morie.fn.kanon.k_anonymity_check`.

    Parameters
    ----------
    X : array-like
        Records; only its length is used, to check against the block.
    quasi_ids : array-like
        Quasi-identifier block, one row per record.
    k : int
        Required anonymity level.

    Returns
    -------
    result : dict
        Keys: estimate, min_class_size, max_class_size, mean_class_size,
        k, satisfies, n_classes, n_violating, n_unique, n.

    References
    ----------
    Sweeney (2002), Int. J. Uncertainty, Fuzziness and Knowledge-Based
    Systems 10(5):557-570, doi:10.1142/S0218488502001648.
    """
    return _impl(X, quasi_ids, k)


def cheatsheet():
    return "dpkb: k-anonymity baseline (NOT DP) -- alias of kanon"


# compact alias per ledger/NAMING.md
kanonymity = k_anonymity
