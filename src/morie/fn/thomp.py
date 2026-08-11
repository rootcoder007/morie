# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thompson sampling for the Beta-Bernoulli bandit."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["thomp", "thompson_sampling"]

_METHOD = "Beta-Bernoulli Thompson sampling"


def thompson_sampling(p, T, alpha0=None, beta0=None, seed=0):
    r"""
    Thompson sampling on a Bernoulli bandit with Beta priors.

    Algorithm 3.2 (BernTS) of Russo et al. (2018): at each period,
    sample theta_k ~ Beta(alpha_k, beta_k) for every action k, apply
    x_t = argmax_k theta_k, observe the Bernoulli reward r_t, and
    conjugately update the chosen action's posterior

        (alpha_x, beta_x) <- (alpha_x + r_t, beta_x + 1 - r_t)

    (their Section 3 update line).  The probability-matching idea
    originates with Thompson (1933).

    Determinism conventions (mirrored bit-exactly in the R arm): each
    period consumes, in order, one Beta draw per action k = 0..K-1 from
    the shared SplitMix64 stream (Marsaglia-Tsang gamma pairs, exactly
    the .ghc_beta1 mirror), then one uniform for the Bernoulli reward
    (r = 1 if u < p[x]).  Argmax ties break to the lowest action.

    Parameters
    ----------
    p : array-like of length K
        True Bernoulli success probabilities of the arms (the simulated
        environment).
    T : int
        Number of periods.
    alpha0, beta0 : array-like of length K, optional
        Beta prior parameters (default all 1: uniform priors).
    seed : int
        SplitMix64 seed.

    Returns
    -------
    result : dict
        Keys: ``estimate`` (0-based arm with the largest posterior mean
        alpha/(alpha+beta) after T periods), ``actions``, ``rewards``,
        ``alpha`` and ``beta`` (posterior parameters), ``post_mean``,
        ``counts``, ``total_reward``, ``method``.

    References
    ----------
    Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I. and Wen, Z.
    (2018). A tutorial on Thompson sampling. Foundations and Trends in
    Machine Learning 11(1), 1-96 (arXiv:1707.02038).  Algorithm 3.2
    (BernTS) and the Section 3 conjugate update.  Local source:
    fetched-wave3/russo-etal-2018-thompson-sampling-tutorial-arxiv1707.02038.pdf.
    Thompson, W. R. (1933). On the likelihood that one unknown
    probability exceeds another in view of the evidence of two samples.
    Biometrika 25(3-4), 285-294.
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    K = len(p)
    for k in range(K):
        if p[k] < 0.0 or p[k] > 1.0:
            raise ValueError("p must lie in [0, 1]")
    T = int(T)
    a = [1.0] * K if alpha0 is None else [float(v) for v in np.atleast_1d(
        np.asarray(alpha0, dtype=float))]
    b = [1.0] * K if beta0 is None else [float(v) for v in np.atleast_1d(
        np.asarray(beta0, dtype=float))]
    if len(a) != K or len(b) != K:
        raise ValueError("alpha0/beta0 must have length K")
    rng = np.random.default_rng(seed)
    actions = np.zeros(T)
    rewards = np.zeros(T)
    counts = [0.0] * K
    for t in range(T):
        best = 0
        besttheta = -1.0
        for k in range(K):
            th = float(rng.beta(a[k], b[k]))
            if th > besttheta:
                besttheta = th
                best = k
        u = float(rng.uniform())
        r = 1.0 if u < p[best] else 0.0
        a[best] += r
        b[best] += 1.0 - r
        counts[best] += 1.0
        actions[t] = float(best)
        rewards[t] = r
    pm = [a[k] / (a[k] + b[k]) for k in range(K)]
    est = 0
    for k in range(1, K):
        if pm[k] > pm[est]:
            est = k
    return RichResult(payload={
        "estimate": float(est),
        "actions": actions,
        "rewards": rewards,
        "alpha": np.asarray(a),
        "beta": np.asarray(b),
        "post_mean": np.asarray(pm),
        "counts": np.asarray(counts),
        "total_reward": float(np.sum(rewards)),
        "method": _METHOD,
    })


thomp = thompson_sampling


def cheatsheet():
    return "thomp(p, T) -> Beta-Bernoulli Thompson sampling (Russo et al 2018, Algorithm 3.2 BernTS)."
