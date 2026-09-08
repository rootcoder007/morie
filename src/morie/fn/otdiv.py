# morie.fn -- function file (rootcoder007/morie)
"""Sinkhorn divergence: the debiased entropic transport cost."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_sinkhorn_divergence"]


def _ot_eps(a, b, C, eps, n_iter):
    T, f, g = ot.sinkhorn(a, b, C, eps, n_iter)
    R = [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]
    return ot.frob(T, C) + eps * ot.kl(T, R)


def ot_sinkhorn_divergence(a, b, Cab, Caa, Cbb, epsilon, max_iter=200):
    """Entropic transport cost with its own bias subtracted off.

    The regularised cost ``OT_eps(mu, mu)`` is not zero, so ``OT_eps``
    alone is not a divergence and a generative model trained on it drifts
    towards a blurred target.  Subtracting half of each self-cost fixes
    that: the result is zero exactly when the two measures coincide, and
    it interpolates between the transport cost (eps -> 0) and an energy
    distance (eps -> infinity).

    Formula: ``S_eps(mu,nu) = OT_eps(mu,nu) - 0.5 (OT_eps(mu,mu) +
    OT_eps(nu,nu))`` with ``OT_eps(mu,nu) = <T*,C> + eps KL(T* | mu (x)
    nu)`` -- Genevay, Peyre & Cuturi (2018) eq. (3)-(4).

    Parameters
    ----------
    a, b : array-like
        The two histograms.
    Cab : array-like, shape (n, m)
        Cross cost.
    Caa : array-like, shape (n, n)
        Cost of ``a`` against itself.
    Cbb : array-like, shape (m, m)
        Cost of ``b`` against itself.
    epsilon : float
        Regularisation strength, positive.
    max_iter : int, default 200
        Sinkhorn sweeps.

    Returns
    -------
    RichResult
        ``S_eps``, ``OT_ab``, ``OT_aa``, ``OT_bb``, ``n``, ``m``.

    References
    ----------
    Genevay, A., Peyre, G. and Cuturi, M. (2018).  Learning generative
    models with Sinkhorn divergences.  Proceedings of Machine Learning
    Research 84:1608-1617 (AISTATS).
    """
    aa = ot.hist(a)
    bb = ot.hist(b)
    Cx = core.mat(Cab)
    Ca = core.mat(Caa)
    Cb = core.mat(Cbb)
    n, m = len(aa), len(bb)
    if len(Cx) != n or len(Cx[0]) != m:
        raise ValueError("cross cost does not match the marginals")
    if len(Ca) != n or len(Cb) != m:
        raise ValueError("self costs do not match the marginals")
    eps = float(epsilon)
    ab = _ot_eps(aa, bb, Cx, eps, max_iter)
    a2 = _ot_eps(aa, aa, Ca, eps, max_iter)
    b2 = _ot_eps(bb, bb, Cb, eps, max_iter)
    return RichResult(payload={
        "S_eps": ab - 0.5 * (a2 + b2), "OT_ab": ab, "OT_aa": a2,
        "OT_bb": b2, "n": n, "m": m,
        "method": "Sinkhorn divergence"})


def cheatsheet():
    return "otdiv: debiased Sinkhorn divergence between two histograms"
