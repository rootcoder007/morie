# morie.fn -- function file (rootcoder007/morie)
"""End-to-end causal pipeline: structure discovery then effect estimation."""

from . import _array_core as np

from ._richresult import RichResult
from .bdcrt import _parse, backdoor_criterion
from .fciag import fci_algorithm

__all__ = ["deci_model"]


def deci_model(data, treatment, outcome, names=None, alpha=0.01, dag=None):
    r"""Discover structure, pick an adjustment set, estimate the effect.

    The "end-to-end" idea behind DECI is that structure learning and
    effect estimation should not be separate manual steps. This is the
    linear-Gaussian, non-neural version of that pipeline:

    1. learn a skeleton with :func:`morie.fn.fciag.fci_algorithm`
       (or accept a supplied ``dag``);
    2. take the treatment's neighbours excluding the outcome as a
       candidate adjustment set, and check it with the back-door
       criterion when a DAG is available;
    3. estimate the effect by OLS of the outcome on treatment plus the
       adjustment set.

    Because the skeleton is undirected, step 2 is a *heuristic* when no
    DAG is given -- the result reports which mode it ran in rather than
    presenting a discovered adjustment set as if it were verified.

    Parameters
    ----------
    data : array-like, shape (n, p)
        Observations.
    treatment, outcome : int or str
        Column index or name.
    names : sequence, optional
        Column names.
    alpha : float, default 0.01
        Discovery test level.
    dag : dict or edge list, optional
        A known graph; when given, the adjustment set is verified by
        the back-door criterion instead of guessed.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``adjustment_set``, ``backdoor_verified``,
        ``discovered_edges``, ``naive`` (unadjusted coefficient),
        ``n``, ``method``.

    References
    ----------
    Geffner, T. et al. (2022). Deep end-to-end causal inference.
    arXiv:2202.02195. (the end-to-end framing; this is the linear
    analogue, not the neural model)

    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Def. 3.3.1 (back-door criterion).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2-D.")
    n, p = X.shape
    labels = list(names) if names is not None else list(range(p))
    if len(labels) != p:
        raise ValueError(f"names has {len(labels)} entries but data has {p} columns.")

    def idx(v):
        if v in labels:
            return labels.index(v)
        if isinstance(v, (int, np.integer)) and 0 <= v < p:
            return int(v)
        raise ValueError(f"{v!r} is not a column of data.")

    ti, oi = idx(treatment), idx(outcome)
    if ti == oi:
        raise ValueError("treatment and outcome must differ.")

    disc = fci_algorithm(X, alpha=alpha, names=labels)
    adjmat = disc["adjacency"]
    cand = [labels[v] for v in range(p) if v not in (ti, oi) and adjmat[ti, v]]

    verified = None
    if dag is not None:
        _, _, nodes = _parse(dag)
        for nm in (labels[ti], labels[oi]):
            if nm not in nodes:
                raise ValueError(f"{nm!r} is not a node of the supplied dag.")
        cand = [c for c in cand if c in nodes]
        verified = bool(backdoor_criterion(dag, labels[ti], labels[oi], Z=tuple(cand))["satisfied"])

    cols = [np.ones(n), X[:, ti]] + [X[:, labels.index(cname)] for cname in cand]
    b, *_ = np.linalg.lstsq(np.column_stack(cols), X[:, oi], rcond=None)
    bn, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X[:, ti]]), X[:, oi], rcond=None)

    return RichResult(
        payload={
            "estimate": float(b[1]),
            "adjustment_set": tuple(cand),
            "backdoor_verified": verified,
            "discovered_edges": disc["edges"],
            "naive": float(bn[1]),
            "n": int(n),
            "method": "End-to-end pipeline: FCI skeleton, adjustment set, OLS effect (linear)",
        }
    )


def cheatsheet():
    return "deciA: discover skeleton, adjust on the treatment's neighbours, OLS the effect"


# compact alias per ledger/NAMING.md
decimodel = deci_model
