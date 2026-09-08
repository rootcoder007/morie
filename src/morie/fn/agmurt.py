# morie.fn -- function file (rootcoder007/morie)
"""MuZero Reanalyze targets and prioritised-replay weights."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mzreanal", "muzero_reanalyze_target"]


def mzreanal(rewards, freshvalues, visits, n=5, gamma=0.997,
             alpha=1.0, beta=1.0, oldvalues=None):
    """Recompute value and policy targets from a fresh search.

    MuZero Reanalyze revisits stored trajectories with the current
    network, re-running the search to obtain fresh root values nu and
    fresh visit counts.  The value target is the same n-step return as in
    ordinary training but bootstrapped from the fresh value,

        z_t = sum_{j<n} gamma^j u_{t+j} + gamma^n nu_{t+n},

    and the policy target is the normalised visit distribution
    pi_t(a) = N_t(a) / sum_b N_t(b).  Replay priority uses the gap
    between the search value and the return,

        p_i = | nu_i - z_i |,  P(i) = p_i^alpha / sum_k p_k^alpha,
        w_i = ( (1/N) (1/P(i)) )^beta.

    Parameters
    ----------
    rewards : array-like
        Environment rewards u_1..u_T.
    freshvalues : array-like
        Root values nu from the fresh search, length T.
    visits : array-like, shape (T, A)
        Fresh root visit counts per action.
    n : int
        Bootstrap horizon.
    gamma : float
        Discount factor.
    alpha, beta : float
        Prioritised-replay exponents; the paper uses alpha = beta = 1.
    oldvalues : array-like or None
        Values used for the priority gap; ``None`` uses ``freshvalues``.

    Returns
    -------
    RichResult
        ``target``, ``policy``, ``priority``, ``prob``, ``weight``, ``T``,
        ``A``, ``n``, ``gamma``.

    References
    ----------
    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
    Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D.,
    Graepel, T., Lillicrap, T. and Silver, D. (2020), "Mastering Atari,
    Go, chess and shogi by planning with a learned model", Nature 588,
    604-609; arXiv:1911.08265.  Read from the ar5iv rendering of the
    arXiv source.  Appendix H describes Reanalyze as re-running MCTS with the latest
    parameters to provide fresh targets; the Training paragraph of the
    Methods gives p_i = |nu_i - z_i|, P(i) = p_i^alpha / sum_k p_k^alpha,
    w_i = ((1/N)(1/P(i)))^beta and alpha = beta = 1.
    """
    u = C.vec(rewards)
    nu = C.vec(freshvalues)
    T = len(u)
    if len(nu) != T:
        raise ValueError("rewards and freshvalues must have the same length")
    Vs = C.mat(visits)
    if len(Vs) != T:
        raise ValueError("visits must have one row per time step")
    A = len(Vs[0])
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
            s += (g ** n) * nu[t + n]
        z.append(s)
    pol = []
    for t in range(T):
        tot = sum(Vs[t])
        pol.append([0.0] * A if tot == 0.0 else [v / tot for v in Vs[t]])
    ov = nu if oldvalues is None else C.vec(oldvalues)
    if len(ov) != T:
        raise ValueError("oldvalues must have length T")
    pr = [abs(ov[t] - z[t]) for t in range(T)]
    a = float(alpha)
    pa = [v ** a for v in pr]
    sp = sum(pa)
    prob = [1.0 / T] * T if sp == 0.0 else [v / sp for v in pa]
    b = float(beta)
    w = [(1.0 / (T * prob[t])) ** b for t in range(T)]
    return RichResult(payload={
        "target": z, "policy": pol, "priority": pr, "prob": prob,
        "weight": w, "T": T, "A": A, "n": n, "gamma": g,
        "method": "MuZero Reanalyze targets (Schrittwieser et al. 2020 App. H)"})


muzero_reanalyze_target = mzreanal


def cheatsheet():
    return "agmurt: MuZero Reanalyze targets and prioritised-replay weights."
