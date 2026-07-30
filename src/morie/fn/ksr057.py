# morie.fn -- function file (rootcoder007/morie)
"""M-estimator asymptotic normality (Kosorok Thm 2.13)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_m_normality", "kosorok_ch2_m_estimator_master_theorem"]


def kosorok_m_normality(m_dot_scores, V=None):
    r"""Asymptotic normality of an M-estimator (Kosorok Thm. 2.13,
    p. 29):

    .. math:: \sqrt n(\hat\theta_n - \theta_0)
              \rightsquigarrow -V^{-1}Z,

    with :math:`Z` the Gaussian limit of
    :math:`\mathbb G_n \dot m_{\theta_0}` and :math:`V` the second
    derivative of the population criterion at :math:`\theta_0`.

    The limit is a SANDWICH, :math:`V^{-1}\Sigma V^{-1}` with
    :math:`\Sigma = P\dot m\dot m'`, and the two matrices are
    different objects: :math:`V` describes the curvature of the
    criterion, :math:`\Sigma` the variability of its gradient. They
    coincide only when the criterion is a correctly specified
    log-likelihood -- the information equality -- and assuming they
    do elsewhere is the standard way to get standard errors wrong.
    Both are returned, along with whether they agree.

    Parameters
    ----------
    m_dot_scores : array-like, shape (n, p)
        The gradient :math:`\dot m_{\theta_0}` at each observation.
    V : array-like, optional
        The curvature matrix; :math:`\Sigma` is used when omitted,
        which ASSUMES the information equality.

    Returns
    -------
    RichResult
        keys: ``Sigma``, ``V``, ``avar`` (the sandwich), ``se``,
        ``information_equality_assumed``, ``information_equality_holds``,
        ``n``, ``p``, ``method``.
    References
    ----------
    Kosorok, Thm. 2.13, p. 29.
    """
    S = np.atleast_2d(np.asarray(m_dot_scores, dtype=float))
    if S.ndim != 2:
        raise ValueError("m_dot_scores must be (n, p).")
    n, p = S.shape
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    Sigma = S.T @ S / n
    assumed = V is None
    Vm = Sigma if assumed else np.atleast_2d(np.asarray(V, dtype=float))
    if Vm.shape != (p, p):
        raise ValueError(f"V must be {p} by {p}, got {Vm.shape}.")
    Vi = np.linalg.pinv(Vm)
    avar = Vi @ Sigma @ Vi
    return RichResult(payload={
        "Sigma": Sigma, "V": Vm, "avar": avar,
        "se": np.sqrt(np.maximum(np.diag(avar), 0.0) / n),
        "information_equality_assumed": assumed,
        "information_equality_holds": bool(np.allclose(Vm, Sigma, rtol=1e-6)),
        "n": int(n), "p": int(p),
        "method": "M-estimator normality (Thm. 2.13); the limit is a SANDWICH, not V^{-1} alone"})


def cheatsheet():
    return "ksr057: V is curvature, Sigma is gradient variance -- equal only for a correct likelihood"


#: Catalogue alias for :func:`kosorok_m_normality`.
kosorok_ch2_m_estimator_master_theorem = kosorok_m_normality
