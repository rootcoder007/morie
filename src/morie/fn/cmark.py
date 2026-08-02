# morie.fn -- function file (rootcoder007/morie)
"""Causal Markov condition: each node independent of non-descendants given parents."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult
from .bdcrt import _descendants, _has_cycle, _parse

__all__ = ["causal_markov_condition"]


def _partial_corr(a, b, C):
    """Partial correlation of a, b given the columns of C (via residuals)."""
    n = a.size
    D = np.column_stack([np.ones(n)] + ([C] if C.size else []))
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def causal_markov_condition(dag, data=None, alpha=0.01):
    r"""Enumerate (and optionally test) the local Markov independencies.

    The causal Markov condition says every node is independent of its
    non-descendants given its parents, equivalently

    .. math:: P(V) = \prod_i P(V_i \mid \mathrm{Pa}(V_i)).

    Returns the full list of implied statements
    ``V _||_ W | Pa(V)`` for each non-descendant non-parent W. With
    ``data`` (a mapping node -> array), each statement is tested by
    the Fisher-z partial correlation test at level ``alpha`` and
    violations are reported -- a linear-Gaussian check, so a violation
    is evidence against the DAG, while passing supports (not proves)
    it.

    Parameters
    ----------
    dag : dict or edge list
        ``{node: [children]}`` or ``[(u, v), ...]``.
    data : mapping, optional
        node -> 1-D array, all the same length.
    alpha : float, default 0.01
        Test level for the Fisher-z checks.

    Returns
    -------
    RichResult
        keys: ``implied`` (list of (V, W, parents) tuples),
        ``n_implied``, ``violations`` (subset with p < alpha; None
        when no data), ``holds`` (None when no data), ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
    (2nd ed.). Cambridge University Press. Sec. 1.4 (Markov
    condition), Thm 1.4.1 (Markov factorisation).
    """
    children, parents, nodes = _parse(dag)
    if _has_cycle(children, nodes):
        raise ValueError("dag contains a directed cycle.")

    implied = []
    for v in sorted(nodes):
        nd = nodes - _descendants(v, children) - {v} - parents[v]
        for w in sorted(nd):
            implied.append((v, w, tuple(sorted(parents[v]))))

    violations = holds = None
    if data is not None:
        arrays = {k: np.asarray(vv, dtype=float).ravel() for k, vv in data.items()}
        missing = nodes - set(arrays)
        if missing:
            raise ValueError(f"data missing nodes: {sorted(missing)}.")
        n = len(next(iter(arrays.values())))
        violations = []
        for v, w, pa in implied:
            C = np.column_stack([arrays[p] for p in pa]) if pa else np.empty((n, 0))
            r = _partial_corr(arrays[v], arrays[w], C)
            k = len(pa)
            if n - k - 3 <= 0:
                raise ValueError("too few observations for the Fisher-z test.")
            r = min(max(r, -0.999999), 0.999999)
            z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
            p = float(2 * stats.norm.sf(abs(z)))
            if p < alpha:
                violations.append({"pair": (v, w), "given": pa, "partial_corr": r, "p_value": p})
        holds = len(violations) == 0

    return RichResult(
        payload={
            "implied": implied,
            "n_implied": len(implied),
            "violations": violations,
            "holds": holds,
            "method": "Causal Markov condition (local independencies, Fisher-z when data given)",
        }
    )


def cheatsheet():
    return "cmark: V _||_ NonDesc(V)\\Pa(V) | Pa(V) for every node; Fisher-z test with data"
