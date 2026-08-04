# morie.fn -- function file (rootcoder007/morie)
"""Besag-York-Mollie disease-mapping log-posterior kernel."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bymfit", "besag_York_Mollie"]


def bymfit(y, E, A, u, v, taus=1.0, tauv=1.0, X=None, beta=None):
    """Evaluate the convolution model's likelihood and its two priors.

    Small-area disease counts are overdispersed relative to Poisson and
    spatially patterned at the same time, so the convolution model splits
    the area effect into a spatially structured part and an unstructured
    part:

        y_i  ~ Poisson( E_i exp( x_i' beta + u_i + v_i ) )
        u    ~ ICAR: p(u) propto exp( -(tau_u/2) sum_{i~j} (u_i - u_j)^2 )
        v_i  ~ N(0, 1/tau_v) independently

    The ICAR prior is improper -- it is invariant to adding a constant to
    every u_i -- so the routine also reports the sum of u, which a fit
    must constrain to zero for the two components to be separately
    identified.

    Parameters
    ----------
    y : array-like
        Observed counts per area.
    E : array-like
        Expected counts (offsets), strictly positive.
    A : array-like, shape (n, n)
        Neighbourhood adjacency; non-zero means adjacent.  Symmetrised,
        and each unordered pair counted once.
    u : array-like
        Spatially structured effects.
    v : array-like
        Unstructured effects.
    taus, tauv : float
        Precisions of the structured and unstructured components.
    X : array-like or None
        Covariates, one row per area.
    beta : array-like or None
        Covariate coefficients.

    Returns
    -------
    RichResult
        ``logpost``, ``loglik``, ``logpu``, ``logpv``, ``rr``, ``fitted``,
        ``usum``, ``npair``, ``n``.

    References
    ----------
    Besag, J., York, J. and Mollie, A. (1991), "Bayesian image
    restoration, with two applications in spatial statistics", Annals of
    the Institute of Statistical Mathematics 43(1), 1-20, which
    introduced the convolution of an intrinsic conditional autoregression
    with independent noise for disease mapping; the intrinsic
    autoregression itself is Besag, J. (1974), Journal of the Royal
    Statistical Society B 36(2), 192-236.  Standard published form;
    neither article was in the local corpus and neither was read for this
    implementation.
    """
    y = C.vec(y)
    E = C.vec(E)
    u = C.vec(u)
    v = C.vec(v)
    n = len(y)
    if len(E) != n or len(u) != n or len(v) != n:
        raise ValueError("y, E, u and v must have the same length")
    if any(t <= 0.0 for t in E):
        raise ValueError("expected counts must be strictly positive")
    if any(t < 0.0 for t in y):
        raise ValueError("counts must be non-negative")
    Am = C.mat(A)
    if len(Am) != n or len(Am[0]) != n:
        raise ValueError("A must be n by n")
    ts, tv = float(taus), float(tauv)
    if ts <= 0.0 or tv <= 0.0:
        raise ValueError("precisions must be strictly positive")
    if X is None:
        eta0 = [0.0] * n
    else:
        Xm = C.mat(X)
        if len(Xm) != n:
            raise ValueError("X must have one row per area")
        b = C.vec(beta)
        if len(b) != len(Xm[0]):
            raise ValueError("beta must have one entry per column of X")
        eta0 = [sum(Xm[i][j] * b[j] for j in range(len(b))) for i in range(n)]
    rr = [math.exp(eta0[i] + u[i] + v[i]) for i in range(n)]
    mu = [E[i] * rr[i] for i in range(n)]
    ll = sum(y[i] * math.log(mu[i]) - mu[i] - math.lgamma(y[i] + 1.0)
             for i in range(n))
    q = 0.0
    npair = 0
    for i in range(n):
        for j in range(i + 1, n):
            if Am[i][j] != 0.0 or Am[j][i] != 0.0:
                q += (u[i] - u[j]) ** 2
                npair += 1
    lpu = 0.5 * (n - 1) * math.log(ts) - 0.5 * ts * q
    lpv = 0.5 * n * math.log(tv) - 0.5 * tv * sum(t * t for t in v) \
        - 0.5 * n * math.log(2.0 * math.pi)
    return RichResult(payload={
        "logpost": ll + lpu + lpv, "loglik": ll, "logpu": lpu,
        "logpv": lpv, "rr": rr, "fitted": mu, "usum": sum(u),
        "npair": npair, "n": n,
        "method": "BYM convolution log-posterior kernel (Besag-York-Mollie 1991)"})


besag_York_Mollie = bymfit


def cheatsheet():
    return "besagl: Besag-York-Mollie disease-mapping log-posterior kernel."
