# morie.fn -- function file (rootcoder007/morie)
"""Back-door adjustment formula (causal effect via covariate adjustment)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["backdoor_adjustment_formula"]


def _key(v):
    """Plain Python scalar for dict keys, so numpy types do not leak out."""
    return v.item() if hasattr(v, "item") else v


def backdoor_adjustment_formula(X, Y, Z, x=None):
    r"""Back-door adjustment: the causal effect by stratifying on Z.

    .. math::

        P(Y = y \mid do(X = x)) = \sum_z P(Y = y \mid X = x, Z = z)\,P(Z = z)

    The adjustment reweights each stratum by how common it is in the
    *whole* population, :math:`P(Z=z)`, rather than by how common it is
    among the treated. That reweighting is the entire content of the
    formula: the conditional is estimated within strata where Z is held
    fixed, then averaged over the population distribution of Z rather
    than the treatment-specific one.

    This is a causal effect only when Z actually satisfies the back-door
    criterion for (X, Y): no member of Z is a descendant of X, and Z
    blocks every path from X to Y that begins with an arrow into X. That
    is a claim about the causal graph, not about the data, and nothing
    here can check it. Adjusting for a collider or a mediator returns a
    number with the same confidence and no causal meaning. Simpson's
    paradox is this failure exactly: the stratified and unstratified
    answers can point in opposite directions, and which one is causal
    depends on the graph, not the arithmetic.

    All three variables are treated as discrete. Continuous inputs would
    need a model for :math:`P(Y \mid X, Z)`, which is a different
    estimator rather than this formula.

    Parameters
    ----------
    X : array-like, shape (n,)
        Treatment, discrete.
    Y : array-like, shape (n,)
        Outcome, discrete.
    Z : array-like, shape (n,) or (n, k)
        Adjustment set, discrete. Several columns are combined into a
        joint stratum label.
    x : scalar, optional
        Treatment value to intervene on. Defaults to every observed
        level, giving the full interventional distribution.

    Returns
    -------
    RichResult
        keys: ``distribution`` (``{x: {y: P(y | do(x))}}``), ``strata``,
        ``p_z``, ``support_x``, ``support_y``, ``n``,
        ``incomplete_strata``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*,
    2nd edn. Cambridge University Press. Theorem 3.3.2, the back-door
    adjustment.
    """
    xa = np.asarray(X).ravel()
    ya = np.asarray(Y).ravel()
    za = np.asarray(Z)
    if za.ndim == 1:
        za = za.reshape(-1, 1)
    if za.ndim != 2:
        raise ValueError(f"Z must be (n,) or (n, k); got shape {za.shape}.")
    n = xa.size
    if not (ya.size == n and za.shape[0] == n):
        raise ValueError(f"X, Y and Z must share a length; got {n}, {ya.size}, {za.shape[0]}.")
    if n == 0:
        raise ValueError("X, Y and Z must not be empty.")

    zlab = np.array(["|".join(map(str, row)) for row in za])
    z_levels, z_counts = np.unique(zlab, return_counts=True)
    p_z = z_counts / n

    sup_x = np.unique(xa)
    sup_y = np.unique(ya)
    targets = sup_x if x is None else np.atleast_1d(x)
    for t in targets:
        if not np.any(xa == t):
            raise ValueError(f"x = {t!r} does not occur in X; the conditional is undefined.")

    dist = {}
    incomplete = []
    for t in targets:
        acc = {_key(v): 0.0 for v in sup_y}
        for zl, pz in zip(z_levels, p_z):
            in_xz = (zlab == zl) & (xa == t)
            m = int(in_xz.sum())
            if m == 0:
                # No unit with this treatment in this stratum, so
                # P(Y | x, z) is undefined. Recorded rather than treated
                # as zero, which would understate the adjusted effect.
                incomplete.append((_key(t), zl))
                continue
            yz = ya[in_xz]
            for yv in sup_y:
                acc[_key(yv)] += pz * float(np.sum(yz == yv)) / m
        dist[_key(t)] = acc

    return RichResult(
        title="Back-door adjustment",
        payload={
            "distribution": dist,
            "strata": list(z_levels),
            "p_z": p_z,
            "support_x": [_key(v) for v in sup_x],
            "support_y": [_key(v) for v in sup_y],
            "n": int(n),
            "incomplete_strata": incomplete,
            "method": "Back-door adjustment (Pearl 2009, Thm 3.3.2), discrete",
        },
    )


def cheatsheet():
    return "bdrj: back-door adjustment formula"
