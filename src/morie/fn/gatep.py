# morie.fn -- function file (rootcoder007/morie)
"""GATE (group average treatment effect): average CATE over subgroups."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gate_estimation"]


def gate_estimation(cate, X=None, group_var=None, n_groups=4, se=None,
                    labels=None):
    r"""Average an estimated CATE within groups, with the caveat attached.

    .. math:: GATE_g = E[\tau(X) \mid X \in G_g].

    The arithmetic is a group mean. What is worth reporting is
    everything around it.

    A GATE computed by SORTING ON THE FITTED :math:`\hat\tau` -- the
    common "top quintile of predicted effect" table -- is biased
    outward. The same noise that put a unit in the top group also
    inflates its estimate, so the extreme groups' means are too
    extreme, and the spread between top and bottom is not an estimate
    of real heterogeneity. When ``group_var`` is omitted and the
    groups are cut from ``cate`` itself, that is exactly what happens
    and ``selection_on_estimate`` says so. Chernozhukov et al.'s
    sorted-group GATE fixes it with sample splitting: cut the groups
    on one half, average on the other. Passing a ``group_var`` built
    from held-out predictions, or from a pre-registered covariate, is
    the version that means something.

    The standard error, when per-unit ``se`` is supplied, treats the
    group members as independent, which is optimistic for a
    forest-derived CATE where nearby units share leaves. It is
    reported as a lower bound rather than as the standard error.

    Parameters
    ----------
    cate : array-like, shape (n,)
        Estimated per-unit effects.
    X : array-like, optional
        Covariates; used only if ``group_var`` names a column index.
    group_var : array-like or int, optional
        Group labels per unit, or a column index into ``X`` to cut
        into ``n_groups`` quantiles. Cut from ``cate`` itself when
        omitted -- see the warning above.
    n_groups : int
        Number of quantile groups when cutting a continuous variable.
    se : array-like, optional
        Per-unit standard errors of ``cate``.
    labels : sequence, optional
        Names for the groups.

    Returns
    -------
    RichResult
        ``gate`` (array in group order), ``groups``, ``n_by_group``,
        ``se`` and ``ci`` (when ``se`` is given), ``spread``,
        ``monotone``, ``selection_on_estimate``, ``difference``
        (top minus bottom) and its ``difference_p``.

    References
    ----------
    Chernozhukov, Demirer, Duflo and Fernández-Val (2018), "Generic
    Machine Learning Inference on Heterogeneous Treatment Effects",
    NBER working paper 24678.
    Athey and Imbens (2016), *PNAS* 113:7353-7360.

    Examples
    --------
    >>> import numpy as np
    >>> tau = np.linspace(-1, 1, 400)
    >>> g = (np.arange(400) >= 200).astype(int)
    >>> out = gate_estimation(tau, group_var=g)
    >>> [round(v, 3) for v in out["gate"]]
    [-0.501, 0.501]
    """
    tau = np.asarray(cate, dtype=float).ravel()
    n = tau.size
    if n < 2:
        raise ValueError("need at least 2 units, got %d." % n)
    sev = None if se is None else np.asarray(se, dtype=float).ravel()
    if sev is not None and sev.size != n:
        raise ValueError(
            "se has %d entries for %d units." % (sev.size, n)
        )

    selection = False
    if group_var is None:
        source = tau
        selection = True
    elif isinstance(group_var, (int, np.integer)):
        if X is None:
            raise ValueError(
                "group_var was given as a column index but X is None."
            )
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        if not 0 <= int(group_var) < Xa.shape[1]:
            raise ValueError(
                "group_var index %d is out of range for X with %d columns."
                % (int(group_var), Xa.shape[1])
            )
        source = Xa[:, int(group_var)]
    else:
        gv = np.asarray(group_var).ravel()
        if gv.size != n:
            raise ValueError(
                "group_var has %d entries for %d units." % (gv.size, n)
            )
        source = gv

    discrete = source.dtype.kind not in "fc" or np.unique(source).size <= max(
        int(n_groups), 2
    )
    if discrete:
        keys = list(dict.fromkeys(np.asarray(source).tolist()))
        idx = [np.asarray(source) == k for k in keys]
    else:
        k = int(n_groups)
        if k < 2:
            raise ValueError("n_groups must be at least 2, got %d." % k)
        edges = np.quantile(source, np.linspace(0, 1, k + 1))
        edges[0] -= 1e-12
        idx, keys = [], []
        for j in range(k):
            m = (source > edges[j]) & (source <= edges[j + 1])
            if m.any():
                idx.append(m)
                keys.append("Q%d" % (j + 1))
    if labels is not None:
        if len(labels) != len(keys):
            raise ValueError(
                "labels has %d entries for %d groups."
                % (len(labels), len(keys))
            )
        keys = list(labels)
    if len(idx) < 2:
        raise ValueError(
            "the grouping produced %d non-empty group(s); at least 2 are "
            "needed." % len(idx)
        )

    gate = np.array([float(tau[m].mean()) for m in idx])
    ns = np.array([int(m.sum()) for m in idx])
    ses = None
    if sev is not None:
        ses = np.array([float(np.sqrt(np.sum(sev[m] ** 2)) / m.sum())
                        for m in idx])

    z = 1.959963984540054
    order = np.argsort(gate)
    lo, hi = int(order[0]), int(order[-1])
    diff = float(gate[hi] - gate[lo])
    diff_p = None
    if ses is not None:
        d_se = float(np.sqrt(ses[hi] ** 2 + ses[lo] ** 2))
        if d_se > 0:
            diff_p = float(math.erfc(abs(diff) / d_se / np.sqrt(2.0)))

    return RichResult(
        payload={
            "gate": gate,
            "estimate": gate,
            "groups": keys,
            "n_by_group": ns,
            "se": ses,
            "ci": (None if ses is None
                   else np.column_stack([gate - z * ses, gate + z * ses])),
            "se_note": (
                None if ses is None else
                "computed as if group members were independent, which they "
                "are not when the CATE came from a forest; read it as a "
                "lower bound"
            ),
            "spread": float(gate.max() - gate.min()),
            "difference": diff,
            "difference_groups": (keys[lo], keys[hi]),
            "difference_p": diff_p,
            "monotone": bool(np.all(np.diff(gate) >= 0)
                             or np.all(np.diff(gate) <= 0)),
            "selection_on_estimate": selection,
            "selection_warning": (
                "the groups were cut from the estimated CATE itself, so the "
                "noise that placed a unit in the top group also inflated its "
                "estimate; the spread between groups overstates the real "
                "heterogeneity. Cut the groups on held-out predictions or a "
                "pre-registered covariate instead"
                if selection else None
            ),
            "n_groups": len(idx),
            "n": int(n),
            "method": "Group average treatment effects from an estimated CATE",
        }
    )


def cheatsheet():
    return (
        "gatep: average a CATE within groups, flagging the outward bias when "
        "the groups are cut from the estimate itself"
    )


# compact alias per ledger/NAMING.md
gateestimation = gate_estimation
