# morie.fn -- function file (rootcoder007/morie)
"""Sobolev (H^-1) approximation to the 1-Wasserstein distance."""

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_sobolev_w1"]


def ot_sobolev_w1(mu, nu, Laplace_inv):
    """Replace W_1 by a Hilbert norm on the difference of the measures.

    ``W_1`` linearised around a common reference measure is exactly the
    ``H^{-1}`` norm of the difference, and unlike ``W_1`` that norm is
    quadratic: it needs one linear solve, not a linear program, and it
    embeds into a Hilbert space so kernel methods apply directly.  The
    price is that it is only a first-order approximation, faithful when
    the two measures are close and unreliable when they are far apart.

    Formula: ``W_1(mu,nu) ~ ||mu - nu||_{H^{-1}} = sqrt((mu-nu)'
    (-Delta)^{-1} (mu-nu))`` -- Peyre (2018), Section 2.

    Parameters
    ----------
    mu, nu : array-like, shape (n,)
        Two measures on the same support.
    Laplace_inv : array-like, shape (n, n)
        Inverse (or pseudo-inverse) of the Laplacian, symmetric positive
        semi-definite.

    Returns
    -------
    RichResult
        ``W1_sob``, ``quad_form``, ``mass_gap``, ``n``.

    References
    ----------
    Peyre, G. (2018).  Comparison between W2 distance and H^-1 norm, and
    localization of Wasserstein distance.  ESAIM: Control, Optimisation
    and Calculus of Variations 24(4):1489-1501.
    doi:10.1051/cocv/2017050.
    """
    a = [float(t) for t in core.vec(mu)]
    b = [float(t) for t in core.vec(nu)]
    L = core.mat(Laplace_inv)
    n = len(a)
    if len(b) != n or len(L) != n or len(L[0]) != n:
        raise ValueError("Laplace_inv must be n by n and match both measures")
    r = [a[i] - b[i] for i in range(n)]
    q = sum(r[i] * L[i][j] * r[j] for i in range(n) for j in range(n))
    if q < 0.0:
        q = 0.0
    return RichResult(payload={
        "W1_sob": q ** 0.5, "quad_form": q,
        "mass_gap": sum(a) - sum(b), "n": n,
        "method": "Sobolev H^-1 approximation to W_1"})


def cheatsheet():
    return "otsobm: H^-1 (Sobolev) approximation to the 1-Wasserstein distance"


# compact alias per ledger/NAMING.md
otsobolevw1 = ot_sobolev_w1
