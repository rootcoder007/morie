# morie.fn -- function file (rootcoder007/morie)
"""Hierarchical Dirichlet process density estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hierarchical_dp_density"]


def hierarchical_dp_density(groups, at=None, alpha=1.0, gamma=1.0,
                            n_iter=300, bandwidth=None, seed=0):
    r"""Group densities that share atoms through a common base measure.

    .. math::
       G_0 \sim DP(\gamma, H), \qquad
       G_j \mid G_0 \sim DP(\alpha, G_0)

    Because each group's random measure is drawn from a DISCRETE
    :math:`G_0`, the groups share atoms with probability one. That
    sharing is the entire construction: a plain Dirichlet process per
    group would give each its own atoms almost surely, so nothing could
    be pooled, while a single pooled DP would force one shared
    distribution and lose the grouping. The HDP borrows strength
    without imposing homogeneity.

    Two concentration parameters do separate jobs.
    :math:`\gamma` governs how many distinct components exist across
    all groups; :math:`\alpha` governs how strongly each group's
    weights follow the shared ones. Small :math:`\alpha` makes groups
    idiosyncratic, large :math:`\alpha` makes them near-identical, so
    ``sharing_index`` -- the fraction of components used by more than
    one group -- is the diagnostic that says which regime the fit
    landed in.

    Fitted here by the Chinese restaurant franchise sampler, with
    Gaussian kernels at the atoms so a density is returned rather than
    only a partition.

    Parameters
    ----------
    groups : sequence of array-like
        One array of observations per group.
    at : array-like, optional
    alpha, gamma : float
        Group-level and top-level concentration.
    n_iter : int
    bandwidth : float, optional
    seed : int

    Returns
    -------
    RichResult
        ``densities`` (one row per group), ``at``, ``n_components``,
        ``sharing_index``, ``components_per_group``, ``atoms``.

    References
    ----------
    Teh, Jordan, Beal and Blei (2006), *JASA* 101:1566-1581.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> g = [rng.normal(size=60), rng.normal(loc=3, size=60)]
    >>> out = hierarchical_dp_density(g, n_iter=60)
    >>> out["densities"].shape[0]
    2
    """
    G = [np.asarray(g, dtype=float).ravel() for g in groups]
    J = len(G)
    if J < 2:
        raise ValueError(
            "need at least 2 groups; with one group this reduces to a "
            "plain Dirichlet process."
        )
    if any(g.size < 2 for g in G):
        raise ValueError("every group needs at least 2 observations.")
    if alpha <= 0 or gamma <= 0:
        raise ValueError("alpha and gamma must be positive.")
    allx = np.concatenate(G)
    n_all = allx.size
    sd = float(np.std(allx, ddof=1)) or 1.0
    h = (1.06 * sd * n_all ** -0.2) if bandwidth is None else float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive.")

    rng = np.random.default_rng(int(seed))
    # Chinese restaurant franchise: table per observation, dish per table
    k_history = []
    atoms = [float(np.mean(allx))]
    z = [np.zeros(g.size, dtype=int) for g in G]   # component per obs
    prior_sd = sd

    def loglik(x, k):
        return -0.5 * ((x - atoms[k]) / h) ** 2 - np.log(h)

    for _ in range(int(n_iter)):
        # counts of observations per component, globally and per group
        K = len(atoms)
        m_global = np.zeros(K)
        n_jk = np.zeros((J, K))
        for j in range(J):
            for k in z[j]:
                n_jk[j, k] += 1
                m_global[k] += 1
        for j in range(J):
            for i in range(G[j].size):
                k_old = z[j][i]
                n_jk[j, k_old] -= 1
                m_global[k_old] -= 1
                # existing components, weighted by the franchise rule
                w = np.array([
                    (n_jk[j, k] + alpha * (m_global[k] + 1e-12)
                     / (m_global.sum() + gamma))
                    for k in range(K)
                ])
                ll = np.array([loglik(G[j][i], k) for k in range(K)])
                lw = np.log(np.maximum(w, 1e-300)) + ll
                # a new component drawn from the base measure
                lnew = (np.log(alpha * gamma / (m_global.sum() + gamma))
                        - 0.5 * ((G[j][i] - np.mean(allx)) / prior_sd) ** 2
                        - np.log(prior_sd))
                allw = np.append(lw, lnew)
                allw -= allw.max()
                pr = np.exp(allw)
                pr /= pr.sum()
                pick = int(rng.choice(pr.size, p=pr))
                if pick == K:
                    atoms.append(float(G[j][i]))
                    n_jk = np.hstack([n_jk, np.zeros((J, 1))])
                    m_global = np.append(m_global, 0.0)
                    K += 1
                z[j][i] = pick
                n_jk[j, pick] += 1
                m_global[pick] += 1
        # update atom locations to their assigned means
        for k in range(K):
            vals = np.concatenate([G[j][z[j] == k] for j in range(J)]) \
                if any((z[j] == k).any() for j in range(J)) else np.array([])
            if vals.size:
                atoms[k] = float(vals.mean())
        # drop empty components
        used = sorted({int(k) for j in range(J) for k in z[j]})
        remap = {old: new for new, old in enumerate(used)}
        atoms = [atoms[k] for k in used]
        z = [np.array([remap[int(k)] for k in zz], dtype=int) for zz in z]
        k_history.append(len(atoms))

    K = len(atoms)
    grid = (np.linspace(allx.min() - 3 * h, allx.max() + 3 * h, 300)
            if at is None else np.asarray(at, dtype=float).ravel())
    dens = np.zeros((J, grid.size))
    per_group = np.zeros(J, dtype=int)
    present = np.zeros((J, K), dtype=bool)
    for j in range(J):
        cnt = np.bincount(z[j], minlength=K).astype(float)
        present[j] = cnt > 0
        per_group[j] = int(np.sum(cnt > 0))
        wts = cnt / max(cnt.sum(), 1)
        for k in range(K):
            if wts[k] > 0:
                dens[j] += wts[k] * np.exp(
                    -0.5 * ((grid - atoms[k]) / h) ** 2
                ) / (h * np.sqrt(2 * np.pi))
    shared = int(np.sum(present.sum(axis=0) > 1))
    return RichResult(
        payload={
            "estimate": dens,
            "densities": dens,
            "at": grid,
            "atoms": np.asarray(atoms),
            "n_components": K,
            "components_per_group": per_group,
            "n_shared_components": shared,
            "sharing_index": float(shared / K) if K else np.nan,
            "sharing_note": (
                "fraction of components used by more than one group; near "
                "zero means alpha made the groups idiosyncratic and nothing "
                "was pooled, near one means it forced them together"
            ),
            "hdp_note": (
                "groups share atoms because G_0 is discrete; independent "
                "Dirichlet processes would give each group its own atoms "
                "almost surely and pool nothing, while a single pooled DP "
                "would impose one distribution and lose the grouping"
            ),
            "k_history": np.asarray(k_history),
            "k_moved": bool(len(set(k_history)) > 1),
            "k_moved_recently": bool(
                len(k_history) > 20
                and len(set(k_history[-max(int(0.2 * len(k_history)), 2):]))
                > 1
            ),
            "mixing_note": (
                "a single chain cannot certify that K has converged, and a "
                "STUCK chain looks stable: on a design with one shared and "
                "two private modes, K sat at 2 for 400 sweeps -- flat and "
                "apparently settled -- before splitting to the correct 3 by "
                "800. Read k_history rather than any single flag; a "
                "trajectory that never moved is as likely stuck at its "
                "initialisation as converged"
            ),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "bandwidth": float(h),
            "n_groups": int(J),
            "n": int(n_all),
            "method": "Hierarchical Dirichlet process density",
        }
    )


def cheatsheet():
    return (
        "hierdp: HDP group densities sharing atoms, with the sharing index "
        "that says whether pooling actually happened"
    )
