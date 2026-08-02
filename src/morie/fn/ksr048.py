# morie.fn -- function file (rootcoder007/morie)
"""Stochastic equicontinuity condition for Z-estimators."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_stochastic_equicontinuity", "kosorok_ch2_z_master_stochastic_equicontinuity"]


def kosorok_stochastic_equicontinuity(psi_n, psi, theta_seq, theta0, n_seq,
                                      grid=None):
    r"""The stochastic equicontinuity condition (Kosorok Eq. 2.12,
    p. 26):

    .. math:: \frac{\big\|\sqrt n(\Psi_n - \Psi)(\theta_n)
              - \sqrt n(\Psi_n - \Psi)(\theta_0)\big\|_L}
              {1 + \sqrt n\,\|\theta_n - \theta_0\|} \to 0.

    The empirical process must not move much between
    :math:`\theta_n` and :math:`\theta_0` -- it is a smoothness
    requirement on the PROCESS, not on any single realisation, and
    it is what allows the increment to be replaced by its limit in
    the linearisation of Eq. (2.13).

    The denominator matters. Without the
    :math:`\sqrt n\|\theta_n-\theta_0\|` term the condition
    would be far too strong to hold at the root-n scale; with it,
    the requirement weakens exactly as fast as
    :math:`\theta_n` approaches :math:`\theta_0`, which is why the
    condition is checkable in practice. ``ratio`` returns the whole
    quantity, and ``numerator`` separately, so the effect of the
    denominator is visible.

    Parameters
    ----------
    psi_n, psi : callable
        ``(theta, t)`` maps as in :mod:`morie.fn.ksr046`.
    theta_seq : sequence
        Candidate parameters, one per entry of ``n_seq``.
    theta0 : object
        True parameter.
    n_seq : sequence of int
        Sample sizes.
    grid : array-like, optional
        Points for the uniform norm.

    Returns
    -------
    RichResult
        keys: ``n``, ``ratio``, ``numerator``, ``denominator``,
        ``holds``, ``denominator_is_essential`` (True), ``method``.
    References
    ----------
    Kosorok, Ch. 2, Eq. (2.12), p. 26.
    """
    g = np.linspace(0.0, 1.0, 51) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    ths = list(theta_seq)
    ns = np.atleast_1d(np.asarray(n_seq, dtype=float)).ravel()
    if len(ths) != ns.size:
        raise ValueError(f"theta_seq has {len(ths)} entries for {ns.size} sizes.")
    if np.any(ns < 1):
        raise ValueError("sample sizes must be at least 1.")
    num, den, rat = [], [], []
    for th, nn in zip(ths, ns):
        a = np.array([float(psi_n(th, v) - psi(th, v)) for v in g])
        b = np.array([float(psi_n(theta0, v) - psi(theta0, v)) for v in g])
        nu = float(np.sqrt(nn) * np.max(np.abs(a - b)))
        de = 1.0 + float(np.sqrt(nn) *
                         np.abs(np.asarray(th, dtype=float) -
                                np.asarray(theta0, dtype=float)).max())
        num.append(nu); den.append(de); rat.append(nu / de)
    return RichResult(payload={
        "n": ns, "ratio": np.array(rat), "numerator": np.array(num),
        "denominator": np.array(den),
        "holds": bool(rat[-1] < rat[0]),
        "denominator_is_essential": True,
        "method": "Stochastic equicontinuity (Eq. 2.12); a condition on the PROCESS, not a realisation"})


def cheatsheet():
    return "ksr048: the 1 + sqrt(n)||theta_n - theta_0|| denominator is what makes it satisfiable"


#: Catalogue alias for :func:`kosorok_stochastic_equicontinuity`.
kosorok_ch2_z_master_stochastic_equicontinuity = kosorok_stochastic_equicontinuity
