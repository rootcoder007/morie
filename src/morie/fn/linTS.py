# morie.fn -- function file (rootcoder007/morie)
"""Linear Thompson sampling for contextual bandits."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lints', 'lin_thompson', 'linthompson']


def lints(contexts, played, rewards, R=0.5, delta=0.1, horizon=None, z=None):
    """Linear Thompson sampling for contextual bandits.

    The posterior over the linear reward parameter is Gaussian with precision B, so one draw from it and a greedy argmax gives the arm. The draw is the only random part, and it is a caller-supplied standard normal vector ``z`` -- passing your own noise is what makes the choice reproducible and lets the R arm return the same arm. The default fills z from a Lehmer minstd stream shared by both languages.


    Formula: B = I_d + sum b b'; muhat = B^-1 f; mutilde ~ N(muhat, v^2 B^-1); a = argmax_i b_i' mutilde

    Parameters
    ----------
    contexts : array-like, shape (n_arms, d)
        Context vector per arm at the current round.
    played : array-like, shape (t, d)
        Context vectors of the arms played so far.
    rewards : array-like, shape (t,)
        Rewards observed for those plays.
    R : float
        Sub-Gaussian parameter of the reward noise.
    delta : float
        Confidence parameter.
    horizon : int, optional
        Time horizon T; the number of plays so far if omitted.
    z : array-like, shape (d,), optional
        Standard normal draw used for the posterior sample.

    Returns
    -------
    RichResult
        ``arm``, ``scores``, ``mu_hat``, ``mu_tilde``, ``v``, ``d``.

    References
    ----------
    Agrawal and Goyal (2013), Thompson Sampling for Contextual Bandits
    with Linear Payoffs, ICML/arXiv:1209.3352.  Algorithm 1 and the
    definition v = R sqrt(9 d ln(T/delta)).  Verified against the paper.
    """
    X = C.mat(contexts)
    P = C.mat(played)
    r = C.vec(rewards)
    d = len(X[0])
    B = C.eye(d)
    f = [0.0] * d
    for i in range(len(P)):
        for a in range(d):
            f[a] += P[i][a] * r[i]
            for b in range(d):
                B[a][b] += P[i][a] * P[i][b]
    Binv = C.inv(B)
    mu = C.matvec(Binv, f)
    T = int(horizon) if horizon is not None else max(len(P), 1)
    v = float(R) * math.sqrt(9.0 * d * math.log(T / float(delta)))
    zz = C.vec(z) if z is not None else None
    if zz is None:
        g = C.Lcg(1)
        zz = [g.norm() for _ in range(d)]
    L = C.chol(Binv)
    mutil = [mu[i] + v * sum(L[i][j] * zz[j] for j in range(d)) for i in range(d)]
    scores = [C.dot(row, mutil) for row in X]
    arm = max(range(len(scores)), key=lambda i: scores[i])
    return RichResult(payload={
        "arm": arm, "scores": scores, "mu_hat": mu, "mu_tilde": mutil,
        "v": v, "d": d, "method": "Linear Thompson sampling (Agrawal-Goyal)"})


lin_thompson = lints
linthompson = lints


def cheatsheet():
    return "linTS: Linear Thompson sampling for contextual bandits."
