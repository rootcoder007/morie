# morie.fn -- function file (rootcoder007/morie)
"""Two-parameter Pitman-Yor weights and partition function."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["poisdir", "ghosal_poisson_dirichlet"]


def poisdir(sigma, M, k, n=None):
    """Pitman-Yor stick-breaking weights and the Gibbs factor V_{n,k}.

    The second parameter is what separates the Pitman-Yor process from
    the Dirichlet process it generalises: sigma = 0 recovers
    DP(M, G) exactly, while sigma > 0 makes the weights decay
    POLYNOMIALLY rather than geometrically, which is why Pitman-Yor is
    the one used when a power-law number of clusters is wanted.

    The weights returned are EXPECTED weights E[W_j], not a draw: the
    stick-breaking factors are independent, so the expectation of the
    product is the product of the expectations, and the result is
    deterministic and identical in both language arms.

    Formula: V_j ~ Beta(1 - sigma, M + j sigma), j = 1, 2, ...;
             W_j = V_j prod_{l<j} (1 - V_l);
             E[V_j] = (1 - sigma)/(M + 1 + (j - 1) sigma);
             V_{n,k} = prod_{i=1}^{k-1} (M + i sigma) / (M + 1)^{[n-1]}

    Parameters
    ----------
    sigma : float
        Discount parameter; here restricted to [0, 1).
    M : float
        Concentration parameter, M > -sigma.
    k : int
        Number of weights returned, k >= 1.
    n : int, optional
        Sample size for the Gibbs factor V_{n,k}; omitted, that factor
        is not computed.

    Returns
    -------
    RichResult
        ``weights`` (E[W_1..W_k]), ``expected_stick`` (E[V_j]),
        ``remaining`` (expected unallocated mass), ``log_Vnk``,
        ``Vnk``, ``sigma``, ``M``, ``k``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Section 14.4 (Pitman-Yor Process), Definition
    14.31: the two-parameter family is the Gibbs process of type sigma
    with V_{n,k} = prod_{i=1}^{k-1}(M + i sigma) / (M + 1)^{[n-1]},
    equation (14.20), the parameters restricted to sigma < 0 with
    M in {-2 sigma, -3 sigma, ...} or sigma in [0, 1) with
    M > -sigma; the weight sequence in size-biased order is the
    Pitman-Yor distribution.  Read from the copy of the book held in
    the corpus.  Only the sigma in [0, 1) branch is implemented; the
    negative-sigma branch is a finite-support family and is refused
    rather than silently mishandled.
    """
    sigma = float(sigma)
    M = float(M)
    k = int(k)
    if not 0.0 <= sigma < 1.0:
        raise ValueError(
            "only the sigma in [0, 1) branch is implemented; the "
            "negative-sigma branch has finite support and is refused")
    if M <= -sigma:
        raise ValueError("M must exceed -sigma")
    if k < 1:
        raise ValueError("k must be at least 1")
    ev = [(1.0 - sigma) / (M + 1.0 + (j - 1) * sigma) for j in range(1, k + 1)]
    w = []
    rest = 1.0
    for j in range(k):
        w.append(rest * ev[j])
        rest *= (1.0 - ev[j])
    lv = float("nan")
    vv = float("nan")
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError("n must be at least 1")
        if k > n:
            raise ValueError("k cannot exceed n")
        num = sum(math.log(M + i * sigma) for i in range(1, k))
        # (M + 1)^{[n-1]} is the rising factorial with n - 1 factors.
        den = sum(math.log(M + 1.0 + i) for i in range(n - 1))
        lv = num - den
        vv = math.exp(lv)
    return RichResult(payload={
        "weights": w, "expected_stick": ev, "remaining": rest,
        "log_Vnk": lv, "Vnk": vv, "sigma": sigma, "M": M, "k": float(k),
        "method": "Pitman-Yor weights and V_{n,k}, Ghosal Definition 14.31"})


ghosal_poisson_dirichlet = poisdir


def cheatsheet():
    return "gh_pd_2param: E[V_j] = (1-sig)/(M+1+(j-1)sig); V_nk eq (14.20)"
