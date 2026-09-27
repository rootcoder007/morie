"""Join count Monte-Carlo (permutation) test."""

from . import _array_core as np
from ._containers import SpatialResult
from ._rng import random_uniform


def _bb(y, W):
    n = len(y)
    return 0.5 * sum(W[i][j] + W[j][i] for i in range(n) for j in range(i + 1, n) if y[i] == 1 and y[j] == 1)


def lacjmc(y_binary, W, nsim=99, *, seed=0):
    """Permutation test of the black-black join count.

    ``BB = (1/2) sum_{i != j} w_ij y_i y_j`` is compared with its values under
    ``nsim`` random relabellings of the map (one Fisher-Yates permutation
    per draw, Philox stream ``s`` for draw ``s``); the upper-tail p-value is
    ``(1 + #{BB_sim >= BB_obs}) / (nsim + 1)``, as in ``spdep::joincount.mc``
    up to its tie handling. The previous body permuted ``y`` independently
    on the two sides of ``y'Wy`` and returned a raw proportion.

    Parameters
    ----------
    y_binary : array-like
        0/1 colour of each unit.
    W : array-like
        Spatial weights (diagonal ignored).
    nsim : int
        Number of permutations.
    seed : int
        Philox key.

    Returns
    -------
    SpatialResult
        ``statistic`` is the observed ``BB``, ``p_value`` the permutation
        p-value, ``expected`` and ``variance`` the mean and variance of the
        permutation distribution; ``extra`` has ``simulated``.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> lacjmc([1, 1, 0, 0], W, nsim=9).statistic
    1.0
    """
    y = [1 if float(v) else 0 for v in np.asarray(y_binary, dtype=float).tolist()]
    Wl = [
        [0.0 if i == j else float(v) for j, v in enumerate(row)]
        for i, row in enumerate(np.asarray(W, dtype=float).tolist())
    ]
    n = len(y)
    obs = _bb(y, Wl)
    sims = []
    for s in range(int(nsim)):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=s)]
        p = list(y)
        for i in range(n - 1, 0, -1):
            j = min(int(u[i] * (i + 1)), i)
            p[i], p[j] = p[j], p[i]
        sims.append(_bb(p, Wl))
    m = sum(sims) / len(sims) if sims else float("nan")
    v = sum((t - m) ** 2 for t in sims) / (len(sims) - 1) if len(sims) > 1 else float("nan")
    return SpatialResult(
        name="lacjmc",
        statistic=obs,
        p_value=(1 + sum(1 for t in sims if t >= obs)) / (len(sims) + 1),
        expected=m,
        variance=v,
        extra={"simulated": sims, "nsim": int(nsim)},
    )


lacjmc_fn = lacjmc


def cheatsheet() -> str:
    return "lacjmc(y, W, nsim) -> join count permutation test (BB)."
