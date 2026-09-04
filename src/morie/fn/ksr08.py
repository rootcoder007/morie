# morie.fn -- function file (rootcoder007/morie)
"""Multiplier bootstrap of the empirical process."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["multboot", "kosorok_multiplier_bootstrap"]


def multboot(x, B=200, seed=1, deterministic_seed=None):
    """Multiplier bootstrap with exponential weights (Dirichlet weights).

    The weights are divided by their own mean, which is what keeps the
    total weight equal to n and makes the multiplier bootstrap
    comparable with the multinomial one; the scaling mu/tau then makes
    the two have the SAME limit.  With standard exponential multipliers
    mu = tau = 1, so the factor is 1 -- but it is written out rather
    than dropped, because it is not 1 for any other weight
    distribution.

    Weights come from a pinned Lehmer generator with a FIXED budget of
    B replicates, so the two language arms agree exactly.

    Formula: Ptilde_n f = n^-1 sum_i (xi_i / xibar_n) f(X_i);
             Gtilde_n = sqrt(n) (mu/tau) (Ptilde_n - P_n),
             xi ~ Exp(1) so mu = tau = 1

    Parameters
    ----------
    x : array-like
        The sample.
    B : int
        Number of replicates (fixed budget).
    seed : int
        Seed for the pinned generator.
    deterministic_seed : int or None, optional
        If given, the pinned generator is seeded from
        SHA-256("ksr08_multiplier:<deterministic_seed>") instead of ``seed``,
        which is the cross-arm reproducible path.

    Returns
    -------
    RichResult
        ``estimate``, ``boot_mean``, ``boot_sd``, ``process_sd``,
        ``ci_lower``, ``ci_upper``, ``mu``, ``tau``, ``B``, ``n``.

    References
    ----------
    Kosorok (2008), Introduction to Empirical Processes and
    Semiparametric Inference, Section 2.2.3: "we can now define a
    multiplier bootstrap empirical measure Ptilde_n f = n^-1 sum
    (xi_i/xibar_n) f(X_i)", with Gtilde_n = sqrt(n)(mu/tau)(Ptilde_n -
    P_n), and the remark that standard exponential multipliers give
    Dirichlet weights.  Fetched as the full text of the book.
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
        # derive the SAME integer from ("ksr08_multiplier", deterministic_seed),
        # so both arms drive the pinned Lcg from an identical start.
        from morie._det_rng import r_seed

        g = C.Lcg(r_seed("ksr08_multiplier", deterministic_seed))
    else:
        g = C.Lcg(seed)
    stats = []
    for _ in range(B):
        w = []
        for _ in range(n):
            u = g.unif()
            if u <= 0.0:
                u = 1e-300
            w.append(-math.log(u))
        wb = sum(w) / n
        if wb == 0.0:
            raise ValueError("the multiplier weights summed to zero")
        stats.append(sum(w[i] / wb * x[i] for i in range(n)) / n)
    bm = sum(stats) / B
    bsd = C.sd(stats, 1)
    q = sorted(stats)
    lo = q[max(0, int(math.floor(0.025 * (B - 1))))]
    hi = q[min(B - 1, int(math.ceil(0.975 * (B - 1))))]
    return RichResult(payload={
        "estimate": Pn, "boot_mean": bm, "boot_sd": bsd,
        "process_sd": math.sqrt(n) * 1.0 * bsd, "ci_lower": lo,
        "ci_upper": hi, "mu": 1.0, "tau": 1.0, "B": float(B),
        "n": float(n),
        "method": "Multiplier bootstrap, Kosorok Section 2.2.3"})


kosorok_multiplier_bootstrap = multboot


def cheatsheet():
    return "ksr08: Ptilde_n f = mean (xi_i/xibar) f(X_i); Exp(1) gives mu=tau=1"
