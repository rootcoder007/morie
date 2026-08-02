# morie.fn -- function file (rootcoder007/morie)
"""Identifiability conditions for causal effects from observational data."""

from . import _array_core as np

from ._richresult import RichResult
from .bdcrt import backdoor_criterion

__all__ = ["identifiability_conditions"]


def identifiability_conditions(dag, X, Y, Z=(), treatment=None, strata=None):
    r"""Check the three textbook identification conditions.

    1. **Exchangeability** given Z -- checkable in the posited graph:
       Z satisfies the back-door criterion for (X, Y).
    2. **Positivity** -- checkable in data: every stratum of Z contains
       both treatment levels. Supply ``treatment`` (binary vector) and
       ``strata`` (vector of stratum labels for the same units).
    3. **Consistency** (Y = Y(T) for the treatment actually received)
       -- a substantive assumption about the treatment's definition;
       reported as such, never as a computed verdict.

    Parameters
    ----------
    dag : dict or edge list
        The causal graph.
    X, Y : hashable
        Treatment and outcome nodes.
    Z : iterable, optional
        Proposed adjustment set.
    treatment : array-like of {0, 1}, optional
        Observed treatment per unit, for the positivity check.
    strata : array-like, optional
        Stratum label per unit (e.g. the discretised Z values).

    Returns
    -------
    RichResult
        keys: ``exchangeability``, ``positivity`` (None without
        data), ``empty_arms`` (strata violating positivity),
        ``consistency`` (always the string "assumption: untestable
        from (Y, T, Z) alone"), ``identifiable`` (None when
        positivity unknown), ``method``.

    References
    ----------
    Hernan, M. A. & Robins, J. M. (2020). *Causal Inference: What If*.
    Chapman & Hall/CRC. Ch. 3 (exchangeability, positivity,
    consistency as the identification triple).
    """
    bd = backdoor_criterion(dag, X, Y, Z=tuple(Z))
    exch = bd["satisfied"]

    positivity = None
    empty = []
    if treatment is not None:
        if strata is None:
            raise ValueError("positivity check needs both treatment and strata.")
        T = np.asarray(treatment, dtype=float).ravel()
        S = np.asarray(strata).ravel()
        if T.size != S.size:
            raise ValueError("treatment and strata must have equal length.")
        if not np.all(np.isin(T, (0.0, 1.0))):
            raise ValueError("treatment must be binary 0/1.")
        for s in np.unique(S):
            arm = T[S == s]
            if arm.min() == arm.max():
                empty.append(s)
        positivity = len(empty) == 0

    identifiable = None if positivity is None else bool(exch and positivity)
    return RichResult(
        payload={
            "exchangeability": bool(exch),
            "positivity": positivity,
            "empty_arms": empty,
            "consistency": "assumption: untestable from (Y, T, Z) alone",
            "identifiable": identifiable,
            "method": "Identification triple: back-door check + stratum positivity + consistency note",
        }
    )


def cheatsheet():
    return "ident: exchangeability (back-door) + positivity (both arms per stratum) + consistency note"
