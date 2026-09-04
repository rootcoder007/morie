# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric bootstrap of the empirical process."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bootemp", "kosorok_bootstrap_empirical"]


def bootemp(x, B=200, seed=1, deterministic_seed=None):
    """Nonparametric (multinomial) bootstrap of the empirical mean process.

    The bootstrap process is sqrt(n)(Phat_n - P_n), NOT
    sqrt(n)(Phat_n - P): it is centred at the EMPIRICAL measure,
    because that is what the resampling actually varies around.
    Centring at P instead is the classic error and inflates the
    variance by the sampling variance of P_n itself.

    The resampling is driven by a pinned Lehmer generator with a FIXED
    budget of B replicates, never a time or convergence criterion, so
    the two language arms draw identical resamples.

    Formula: Phat_n f = n^-1 sum_i W_i f(X_i), (W_1..W_n) ~ Multinomial(n, 1/n);
             Ghat_n = sqrt(n) (Phat_n - P_n)

    Parameters
    ----------
    x : array-like
        The sample.
    B : int
        Number of bootstrap replicates (fixed budget).
    seed : int
        Seed for the pinned generator.
    deterministic_seed : int or None, optional
        If given, the pinned generator is seeded from
        SHA-256("ksr07_bootstrap:<deterministic_seed>") instead of ``seed``,
        which is the cross-arm reproducible path.

    Returns
    -------
    RichResult
        ``estimate`` (P_n f), ``boot_mean``, ``boot_sd``,
        ``process_sd`` (sd of Ghat_n), ``ci_lower``, ``ci_upper``
        (percentile), ``B``, ``n``.

    References
    ----------
    Kosorok (2008), Introduction to Empirical Processes and
    Semiparametric Inference, Section 2.2.3, which defines the
    nonparametric bootstrap empirical measure with multinomial weights
    and the bootstrapped process Ghat_n = sqrt(n)(Phat_n - P_n), and
    Theorem 2.6 relating its convergence to F being P-Donsker.  Fetched
    as the full text of the book.
    """
    x = C.vec(x)
    n = len(x)
    B = int(B)
    if n < 2:
        raise ValueError("the sample must have at least two observations")
    if B < 2:
        raise ValueError("B must be at least 2")
    Pn = sum(x) / n
    if deterministic_seed is not None:
        # SHA-keyed seed: morie._det_rng.r_seed and R's morie_det_rng
        # derive the SAME integer from ("ksr07_bootstrap", deterministic_seed),
        # so both arms drive the pinned Lcg from an identical start.
        from morie._det_rng import r_seed

        g = C.Lcg(r_seed("ksr07_bootstrap", deterministic_seed))
    else:
        g = C.Lcg(seed)
    stats = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            j = int(g.unif() * n)
            if j >= n:
                j = n - 1
            s += x[j]
        stats.append(s / n)
    bm = sum(stats) / B
    bsd = C.sd(stats, 1)
    q = sorted(stats)
    lo = q[max(0, int(math.floor(0.025 * (B - 1))))]
    hi = q[min(B - 1, int(math.ceil(0.975 * (B - 1))))]
    return RichResult(payload={
        "estimate": Pn, "boot_mean": bm, "boot_sd": bsd,
        "process_sd": math.sqrt(n) * bsd, "ci_lower": lo, "ci_upper": hi,
        "B": float(B), "n": float(n),
        "method": "Nonparametric bootstrap, Kosorok Section 2.2.3"})


kosorok_bootstrap_empirical = bootemp


def cheatsheet():
    return "ksr07: Ghat_n = sqrt(n)(Phat_n - P_n), centred at P_n not P"
