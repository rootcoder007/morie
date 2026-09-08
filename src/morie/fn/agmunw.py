# morie.fn -- function file (rootcoder007/morie)
"""MuZero n-step bootstrapped value targets."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mznstep", "muzero_n_step_value"]


def mznstep(rewards, values, n=5, gamma=0.997):
    """n-step discounted return targets bootstrapped from search values.

    For a trajectory with environment rewards u_1, u_2, ... and MCTS root
    values nu_1, nu_2, ..., the training target for time t is the sum of
    the next n discounted rewards plus the discounted search value n steps
    ahead:

        z_t = u_{t+1} + gamma u_{t+2} + ... + gamma^{n-1} u_{t+n}
              + gamma^n nu_{t+n}

    Past the end of the trajectory the bootstrap term is dropped and the
    remaining rewards are summed, which is the same expression with the
    truncated tail.  This is the trajectory-level counterpart of the
    within-search backup G^k of Equation (3).

    Parameters
    ----------
    rewards : array-like
        Environment rewards u_1..u_T, one per transition.
    values : array-like
        Search values nu_1..nu_T at the same time steps.
    n : int
        Bootstrap horizon, >= 1.
    gamma : float
        Discount factor.

    Returns
    -------
    RichResult
        ``target``, ``T``, ``n``, ``gamma``, ``mean``.

    References
    ----------
    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
    Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D.,
    Graepel, T., Lillicrap, T. and Silver, D. (2020), "Mastering Atari,
    Go, chess and shogi by planning with a learned model", Nature 588,
    604-609; arXiv:1911.08265.  Read from the ar5iv rendering of the
    arXiv source.  The n-step return z and the search value nu are named in the
    Training paragraph of the Methods (priority p_i = |nu_i - z_i|); the
    within-search form G^k = sum_tau gamma^tau r_{k+1+tau} + gamma^{l-k}
    v^l is their Equation (3), of which this is the trajectory version.
    """
    u = C.vec(rewards)
    v = C.vec(values)
    T = len(u)
    if len(v) != T:
        raise ValueError("rewards and values must have the same length")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    g = float(gamma)
    z = []
    for t in range(T):
        s = 0.0
        for j in range(n):
            if t + j < T:
                s += (g ** j) * u[t + j]
        if t + n < T:
            s += (g ** n) * v[t + n]
        z.append(s)
    return RichResult(payload={
        "target": z, "T": T, "n": n, "gamma": g,
        "mean": sum(z) / T if T else float("nan"),
        "method": "MuZero n-step value target (Schrittwieser et al. 2020)"})


muzero_n_step_value = mznstep


def cheatsheet():
    return "agmunw: MuZero n-step bootstrapped value targets."
