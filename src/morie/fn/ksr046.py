# morie.fn -- function file (rootcoder007/morie)
"""Z-estimator consistency theorem (Kosorok Thm 2.10)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_z_consistency", "kosorok_ch2_z_estimator_consistency"]


def kosorok_z_consistency(psi_n, psi, theta_seq, theta0, grid=None):
    r"""Z-estimator consistency (Kosorok Thm. 2.10, p. 24):

    if :math:`\sup_\theta \|\Psi_n(\theta) - \Psi(\theta)\|_L
    \to 0` and :math:`\|\Psi_n(\theta_n)\|_L \to 0`, then
    :math:`\theta_n \to \theta_0`.

    Two conditions, and NEITHER alone suffices. The first is uniform
    convergence of the empirical map -- a Glivenko-Cantelli
    statement, which is where empirical process theory enters. The
    second only says :math:`\theta_n` nearly solves the equation;
    with a badly behaved :math:`\Psi` a near-root can sit far from
    the true root, so the uniform convergence is what rules that out.

    Note the norm is :math:`\|\cdot\|_L` on a function space, not a
    Euclidean norm: the theorem covers infinite-dimensional
    parameters such as the survival function of Eq. (2.11), which is
    the reason it is stated this way.

    Both conditions are evaluated along the supplied sequence, so
    the theorem is exercised rather than described.

    Parameters
    ----------
    psi_n : callable
        ``psi_n(theta, t)``, the empirical map.
    psi : callable
        ``psi(theta, t)``, its limit.
    theta_seq : sequence
        The candidate sequence, increasing in n.
    theta0 : object
        The true parameter.
    grid : array-like, optional
        Points at which the uniform norm is taken.

    Returns
    -------
    RichResult
        keys: ``sup_differences``, ``near_root_norms``,
        ``theta_distances``, ``uniform_convergence``,
        ``near_root``, ``consistent``, ``both_needed`` (True),
        ``method``.
    References
    ----------
    Kosorok, Thm. 2.10, p. 24.
    """
    g = np.linspace(0.0, 1.0, 51) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    seq = list(theta_seq)
    if len(seq) < 2:
        raise ValueError("need at least 2 elements in theta_seq to see a trend.")
    sup_d, root_n, dist = [], [], []
    for th in seq:
        a = np.array([float(psi_n(th, v)) for v in g])
        b = np.array([float(psi(th, v)) for v in g])
        sup_d.append(float(np.max(np.abs(a - b))))
        root_n.append(float(np.max(np.abs(a))))
        dist.append(float(np.abs(np.asarray(th, dtype=float) -
                                 np.asarray(theta0, dtype=float)).max()))
    uc = bool(sup_d[-1] < sup_d[0])
    nr = bool(root_n[-1] < root_n[0])
    return RichResult(payload={
        "sup_differences": np.array(sup_d),
        "near_root_norms": np.array(root_n),
        "theta_distances": np.array(dist),
        "uniform_convergence": uc, "near_root": nr,
        "consistent": bool(uc and nr and dist[-1] < dist[0]),
        "both_needed": True,
        "norm": "uniform on a function space, not Euclidean",
        "method": "Z-estimator consistency (Thm. 2.10); uniform convergence AND a near-root, together"})


def cheatsheet():
    return "ksr046: a near-root alone proves nothing -- uniform convergence rules out distant near-roots"


#: Catalogue alias for :func:`kosorok_z_consistency`.
kosorok_ch2_z_estimator_consistency = kosorok_z_consistency
