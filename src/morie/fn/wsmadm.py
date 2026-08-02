# morie.fn -- function file (rootcoder007/morie)
"""Admissibility of decision rules by risk dominance."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_admissible"]


def wasserman_admissible(risk, names=None, tol=1e-12):
    r"""Admissibility of decision rules, decided by dominance.

    A rule :math:`T` is INADMISSIBLE when some other rule
    :math:`T'` dominates it:

    .. math:: R(T', F) \le R(T, F)\ \text{for all } F,
              \quad\text{and}\quad
              R(T', F) < R(T, F)\ \text{for some } F,

    and admissible when no such :math:`T'` exists. This is a
    statement about a COLLECTION of rules and a set of states, so it
    cannot be evaluated for one rule in isolation -- the argument
    here is the whole risk table, rules by states.

    Two things follow from the definition that are easy to get
    wrong and are asserted in the tests rather than assumed.
    Admissibility is not optimality: a rule can be admissible purely
    because it is very good at one absurd state and nothing else
    beats it there, and constant rules are the standard example.
    And the strictness matters -- two rules with identical risk
    everywhere do NOT dominate each other, so ties leave both
    admissible.

    Whether the conclusion means anything depends entirely on the
    supplied risk table being the real risk over the real state
    space. A rule admissible against three sampled states may be
    inadmissible against the full family.

    **Source note.** No text in this repository's reference library
    covers statistical decision theory, so unlike its neighbours this
    module carries no page-level citation. The definition
    implemented above is the standard one and is stated in full
    precisely so that it can be checked against any decision-theory
    reference the reader has to hand; the concept is due to Wald's
    decision-theoretic programme.

    Parameters
    ----------
    risk : array-like, shape (n_rules, n_states)
        ``risk[i, j]`` is the risk of rule ``i`` at state ``j``.
        Lower is better.
    names : sequence of str, optional
        Labels for the rules.
    tol : float, default 1e-12
        Comparisons are made up to this tolerance, so that two rules
        agreeing to floating-point noise count as tied rather than
        one dominating the other.

    Returns
    -------
    RichResult
        keys: ``admissible`` (boolean array), ``bool`` (True when
        every rule is admissible), ``dominated_by``,
        ``admissible_names``, ``n_rules``, ``n_states``,
        ``minimax_rule``, ``is_complete_class``, ``method``.
    """
    R = np.atleast_2d(np.asarray(risk, dtype=float))
    if R.ndim != 2:
        raise ValueError("risk must be a 2-D table of rules by states.")
    m, s = R.shape
    if m < 1 or s < 1:
        raise ValueError(f"risk must be non-empty, got shape {(m, s)}.")
    if not np.all(np.isfinite(R)):
        raise ValueError("every risk must be finite to compare rules.")
    if names is not None:
        names = list(names)
        if len(names) != m:
            raise ValueError(
                f"names has {len(names)} entries for {m} rules.")

    adm = np.ones(m, dtype=bool)
    dominated_by = {}
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            # j dominates i: never worse anywhere, strictly better somewhere
            if (np.all(R[j] <= R[i] + tol)
                    and np.any(R[j] < R[i] - tol)):
                adm[i] = False
                dominated_by.setdefault(
                    names[i] if names else i, []).append(
                        names[j] if names else j)
    worst = R.max(axis=1)
    return RichResult(payload={
        "admissible": adm,
        "bool": bool(np.all(adm)),
        "dominated_by": dominated_by,
        "admissible_names": ([names[i] for i in range(m) if adm[i]]
                             if names else np.flatnonzero(adm)),
        "n_rules": int(m), "n_states": int(s),
        "minimax_rule": (names[int(np.argmin(worst))] if names
                         else int(np.argmin(worst))),
        "minimax_risk": float(worst.min()),
        "is_complete_class": bool(np.all(adm)),
        "definition": "T is inadmissible when some T' has R(T',F) <= R(T,F) "
                      "for all F and R(T',F) < R(T,F) for some F",
        "ties_note": "two rules with identical risk everywhere do NOT "
                     "dominate each other; both stay admissible",
        "scope_note": "admissibility is relative to the supplied rules and "
                      "states; a rule admissible against three sampled "
                      "states may be inadmissible against the full family",
        "method": "Admissibility by pairwise risk dominance"})


def cheatsheet():
    return "wsmadm: needs the whole risk table -- admissibility is not a property of one rule alone"
