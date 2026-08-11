"""Parallel tempering MCMC (replica exchange)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ptmcmc", "parallel_tempering"]


def ptmcmc(log_p, temperatures, x0, n_iter=1000, step=1.0, seed=0,
           swap_every=1):
    """
    Parallel tempering: K replicas at temperatures T_1 < ... < T_K,
    each running random-walk Metropolis on pi_k(x) proportional to
    exp(log_p(x) / T_k), with neighbour swap proposals accepted with

        A = min(1, exp[ (1/T_i - 1/T_j) (E(x_i) - E(x_j)) ])

    where E = -log_p (source eq. (4): p = min(1, e^{-(b_i - b_j)(E_j - E_i)})
    with b = 1/T; equivalently the Hukushima-Nemoto exchange
    probability). Swap pairs (k, k+1) are attempted left to right every
    `swap_every` sweeps.

    Per sweep and replica the update consumes exactly one normal (two
    uniforms) for the proposal and one uniform for the accept test;
    each swap attempt consumes one uniform -- the identical order is
    mirrored in the R arm so the chains agree draw for draw.

    Sources
    -------
    Earl, D. J. & Deem, M. W. (2005). Parallel tempering: theory,
    applications, and new perspectives. *PCCP*, 7, 3910-3916,
    arXiv:physics/0508111, eq. (4)
    (fetched-wave3/earl-deem-2005-parallel-tempering.pdf).
    Hukushima, K. & Nemoto, K. (1996). Exchange Monte Carlo method and
    application to spin glass simulations. *J. Phys. Soc. Japan*,
    65(6), 1604-1608, eq. (2.4)
    (fetched-wave3/hukushima-nemoto-1996-exchange-mc.pdf).
    Metropolis, N. et al. (1953). Equation of state calculations by
    fast computing machines. *J. Chemical Physics*, 21, 1087-1092
    (the within-replica walk).

    Parameters
    ----------
    log_p : callable
        Log target density (the T = 1 target), up to a constant.
    temperatures : sequence of float
        Ascending temperatures; the first should be 1.0 for the cold
        chain to target log_p itself.
    x0 : float
        Common scalar starting state.
    n_iter : int
        Sweeps.
    step : float
        Random-walk proposal sd (scaled by sqrt(T_k) per replica).
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).
    swap_every : int
        Attempt neighbour swaps every this many sweeps.

    Returns
    -------
    RichResult
        Keys: chain (cold-chain trace), chains_last (final states),
        accept_rate (per replica), swap_accept_rate (per pair).
    """
    temps = [float(t) for t in temperatures]
    K = len(temps)
    if K < 2:
        raise ValueError("need at least two temperatures")
    if any(b <= 0 for b in temps) or any(temps[i] >= temps[i + 1] for i in range(K - 1)):
        raise ValueError("temperatures must be positive and ascending")
    n_iter = int(n_iter)
    rng = np.random.default_rng(seed)
    x = [float(x0)] * K
    lp = [float(log_p(v)) for v in x]
    acc = [0] * K
    swap_try = [0] * (K - 1)
    swap_acc = [0] * (K - 1)
    cold = []
    for sweep in range(n_iter):
        for k in range(K):
            prop = x[k] + step * np.sqrt(temps[k]) * rng.standard_normal()
            lpp = float(log_p(prop))
            u = rng.uniform(0.0, 1.0)
            if np.log(u) < (lpp - lp[k]) / temps[k]:
                x[k] = prop
                lp[k] = lpp
                acc[k] += 1
        if (sweep + 1) % int(swap_every) == 0:
            for k in range(K - 1):
                swap_try[k] += 1
                delta = (1.0 / temps[k] - 1.0 / temps[k + 1]) * (lp[k + 1] - lp[k])
                u = rng.uniform(0.0, 1.0)
                if np.log(u) < min(0.0, delta):
                    x[k], x[k + 1] = x[k + 1], x[k]
                    lp[k], lp[k + 1] = lp[k + 1], lp[k]
                    swap_acc[k] += 1
        cold.append(x[0])
    return RichResult(payload={
        "chain": cold, "chains_last": list(x),
        "accept_rate": [a / float(n_iter) for a in acc],
        "swap_accept_rate": [
            (swap_acc[k] / float(swap_try[k])) if swap_try[k] else np.nan
            for k in range(K - 1)
        ],
        "temperatures": temps, "n_iter": n_iter, "seed": int(seed),
        "method": "Parallel tempering (Earl-Deem 2005 eq. 4; Hukushima-Nemoto 1996)",
    })


# long descriptive alias (stub-era name)
parallel_tempering = ptmcmc


def cheatsheet():
    return "ptmcmc: replica exchange, swap prob min(1, exp[(1/Ti - 1/Tj)(Ej - Ei)])"
