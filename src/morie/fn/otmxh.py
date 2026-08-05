# morie.fn -- function file (rootcoder007/morie)
"""Wasserstein-type distance between Gaussian mixtures."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_mixture_w2"]


def ot_mixture_w2(mus1, Sigmas1, w1, mus2, Sigmas2, w2):
    """Transport between mixtures, restricted to plans that stay Gaussian.

    The true ``W_2`` between two Gaussian mixtures has no closed form and
    its optimal plan splits components, so the geodesic leaves the mixture
    family altogether.  Restricting the couplings to those that move whole
    components onto whole components keeps everything inside the family
    and reduces the problem to a small discrete transport whose ground
    cost is the closed-form Gaussian ``W_2^2``.  The restriction can only
    raise the cost, so ``MW_2 >= W_2`` always.

    Formula: ``MW_2^2 = min_{w in Pi(p,q)} sum_kl w_kl W_2^2(N(m_k,S_k),
    N(m'_l,S'_l))`` -- Delon & Desolneux (2020) Definition 4.1.

    Parameters
    ----------
    mus1 : array-like, shape (K1, d)
        Component means of the first mixture.
    Sigmas1 : sequence of K1 arrays, each (d, d)
        Component covariances of the first mixture.
    w1 : array-like, shape (K1,)
        Mixture weights, rescaled to sum to one.
    mus2, Sigmas2, w2
        The same for the second mixture.

    Returns
    -------
    RichResult
        ``MW2``, ``MW2_sq``, ``T`` (the component-level plan), ``C``
        (the pairwise Gaussian costs), ``K1``, ``K2``, ``d``.

    References
    ----------
    Delon, J. and Desolneux, A. (2020).  A Wasserstein-type distance in
    the space of Gaussian mixture models.  SIAM Journal on Imaging
    Sciences 13(2):936-970.  doi:10.1137/19M1301047.
    """
    M1 = core.mat(mus1)
    M2 = core.mat(mus2)
    S1 = [core.mat(s) for s in Sigmas1]
    S2 = [core.mat(s) for s in Sigmas2]
    p = ot.hist(w1, normalise=True)
    q = ot.hist(w2, normalise=True)
    K1, K2 = len(M1), len(M2)
    d = len(M1[0])
    if len(M2[0]) != d:
        raise ValueError("the two mixtures must live in the same dimension")
    if len(S1) != K1 or len(S2) != K2 or len(p) != K1 or len(q) != K2:
        raise ValueError("means, covariances and weights must agree in count")
    C = [[ot.w2gauss(M1[k], S1[k], M2[l], S2[l]) for l in range(K2)]
         for k in range(K1)]
    T, cost = ot.emd(p, q, C)
    return RichResult(payload={
        "MW2": cost ** 0.5, "MW2_sq": cost, "T": T, "C": C,
        "K1": K1, "K2": K2, "d": d,
        "method": "Mixture Wasserstein distance MW2"})


def cheatsheet():
    return "otmxh: Wasserstein-type MW2 distance between Gaussian mixtures"


# compact alias per ledger/NAMING.md
otmixturew2 = ot_mixture_w2
