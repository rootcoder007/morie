# morie.fn -- function file (rootcoder007/morie)
"""Fast Causal Inference (FCI) skeleton + orientation with hidden confounders."""

import itertools

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["fci_algorithm"]


def _ci_test(data, i, j, S, alpha):
    """Fisher-z partial-correlation independence test."""
    n = data.shape[0]
    k = len(S)
    if n - k - 3 <= 0:
        return True, 1.0  # no power: treat as independent
    D = np.column_stack([np.ones(n)] + [data[:, s] for s in S])

    def resid(v):
        b, *_ = np.linalg.lstsq(D, v, rcond=None)
        return v - D @ b

    ri, rj = resid(data[:, i]), resid(data[:, j])
    den = np.sqrt((ri**2).sum() * (rj**2).sum())
    r = float((ri * rj).sum() / den) if den > 0 else 0.0
    r = min(max(r, -0.999999), 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    p = float(2 * stats.norm.sf(abs(z)))
    return p >= alpha, p


def fci_algorithm(data, alpha=0.01, max_cond=3, names=None):
    r"""FCI: PAG skeleton and v-structure orientation under latent confounding.

    Runs the PC-style adjacency phase -- remove the edge i-j whenever
    some conditioning set S (searched by increasing size) makes them
    conditionally independent, recording S as the separating set --
    then orients unshielded triples i *-* k *-* j as colliders
    (:math:`i \ast\!\!\to k \leftarrow\!\ast j`) exactly when k is not
    in sepset(i, j).

    FCI differs from PC in what the output *means*: an edge here is a
    PAG edge with circle endpoints, admitting a latent common cause,
    so ``i o-o j`` says "adjacent, orientation undetermined" rather
    than "i causes j". Only the collider orientations are asserted;
    the further FCI orientation rules (R1-R10) are not applied, and
    the result says so.

    Parameters
    ----------
    data : array-like, shape (n, p)
        Observations, one column per variable.
    alpha : float, default 0.01
        Independence-test level.
    max_cond : int, default 3
        Largest conditioning set searched.
    names : sequence, optional
        Variable names for the reported edges.

    Returns
    -------
    RichResult
        keys: ``adjacency`` (p, p) boolean skeleton, ``edges`` (list of
        name pairs), ``sepsets`` (dict), ``colliders`` (list of
        (i, k, j) triples oriented into k), ``orientation_complete``
        (always False -- R1-R10 not applied), ``n``, ``method``.

    References
    ----------
    Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation,
    Prediction, and Search* (2nd ed.). MIT Press. Ch. 6 (FCI and
    partial ancestral graphs).

    Zhang, J. (2008). On the completeness of orientation rules for
    causal discovery in the presence of latent confounders and
    selection bias. *Artificial Intelligence*, 172(16-17), 1873-1896.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2-D (n observations x p variables).")
    n, p = X.shape
    if p < 3:
        raise ValueError(f"need at least 3 variables, got {p}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    mc = int(max_cond)
    if mc < 0:
        raise ValueError("max_cond must be nonnegative.")
    labels = list(names) if names is not None else list(range(p))
    if len(labels) != p:
        raise ValueError(f"names has {len(labels)} entries but data has {p} columns.")

    adj = ~np.eye(p, dtype=bool)
    sep = {}
    for k in range(mc + 1):
        for i, j in itertools.combinations(range(p), 2):
            if not adj[i, j]:
                continue
            others = [v for v in range(p) if v not in (i, j) and (adj[i, v] or adj[j, v])]
            if len(others) < k:
                continue
            for S in itertools.combinations(others, k):
                indep, _ = _ci_test(X, i, j, S, alpha)
                if indep:
                    adj[i, j] = adj[j, i] = False
                    sep[(i, j)] = sep[(j, i)] = S
                    break

    colliders = []
    for kk in range(p):
        nb = [v for v in range(p) if adj[v, kk]]
        for i, j in itertools.combinations(nb, 2):
            if adj[i, j]:
                continue  # shielded
            S = sep.get((i, j))
            if S is not None and kk not in S:
                colliders.append((labels[i], labels[kk], labels[j]))

    edges = [(labels[i], labels[j]) for i, j in itertools.combinations(range(p), 2) if adj[i, j]]
    return RichResult(
        payload={
            "adjacency": adj,
            "edges": edges,
            "sepsets": {(labels[a], labels[b]): tuple(labels[s] for s in S) for (a, b), S in sep.items()},
            "colliders": colliders,
            "orientation_complete": False,
            "n": int(n),
            "method": "FCI skeleton + v-structure orientation (PAG; R1-R10 not applied)",
        }
    )


def cheatsheet():
    return "fciag: PC-style skeleton with sepsets, then unshielded-collider orientation"
