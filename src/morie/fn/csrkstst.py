# morie.fn -- function file (rootcoder007/morie)
"""KS test of nearest-neighbour distance against complete spatial randomness."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["kstest_csr"]


def _nn_distances(P):
    """Nearest-neighbour distance for each point of an (n, d) pattern."""
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _empirical_G(nn, grid):
    r"""\hat G(y) = #(y_i <= y) / n, evaluated on ``grid``."""
    return np.searchsorted(np.sort(nn), grid, side="right") / nn.size


def _as_window(window, P):
    """Normalise ``window`` to (lo, hi) bounds arrays of length d."""
    d = P.shape[1]
    if window is None:
        return P.min(axis=0), P.max(axis=0)
    # One convention only: `d` rows of (min, max), or the same numbers
    # flat. Reading a (2, d) array as lower/upper corners would be just as
    # natural, but in the common case d = 2 both readings have shape
    # (2, 2) and mean different regions, so supporting them together makes
    # a silent misinterpretation possible. Ranges win because they stay
    # unambiguous at every d.
    W = np.asarray(window, dtype=float)
    if W.size != 2 * d:
        raise ValueError(f"window must give {d} (min, max) pairs for {d}-dimensional coords; got shape {W.shape}.")
    pair = W.reshape(d, 2)
    lo, hi = pair[:, 0], pair[:, 1]
    if np.any(hi <= lo):
        raise ValueError("window upper bounds must exceed lower bounds.")
    return lo, hi


def kstest_csr(coords, window=None, cdf=None, nsim=199, seed=None):
    r"""Test a point pattern against complete spatial randomness.

    Compares the empirical nearest-neighbour distance distribution

    .. math:: \hat G(y) = \frac{\#(y_i \le y)}{n}

    against complete spatial randomness through the supremum distance of
    a Kolmogorov-Smirnov statistic.

    The reference distribution is obtained by simulation rather than in
    closed form. Under CSR on the whole plane the nearest-neighbour
    distance would follow :math:`G(y) = 1 - \exp(-\lambda\pi y^2)`, but a
    real pattern is observed inside a bounded window, and points near the
    edge have no neighbours beyond it. Their nearest-neighbour distances
    are biased upward and the closed form no longer holds. Schabenberger
    & Gotway put it directly: the comparison is against the theoretical
    :math:`G(h)` "or, if :math:`G(h)` is not attainable, against the
    average empirical distribution function from the simulation".
    Simulating inside the same window makes the edge effect common to
    the data and to the reference, so it cancels.

    The p-value is the rank of the observed statistic among the
    simulated ones,

    .. math:: p = \frac{1 + \#\{D_i \ge D_{obs}\}}{1 + n_{sim}}

    which is the convention behind the same authors' worked example:
    with 200 simulations and none exceeding the observed value, they
    report p = 0.00498 = 1/201.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Event locations. A one-dimensional input is read as n points on
        a line.
    window : array-like, optional
        Observation region, as ``d`` (min, max) pairs -- shape ``(d, 2)``
        or a flat sequence of length ``2 * d``. Defaults to
        the bounding box of ``coords``. Give it explicitly when the true
        region is larger than the observed extent: the bounding box of
        the data is itself slightly too small, which inflates the
        simulated intensity.
    cdf : callable, optional
        Analytic null CDF of the nearest-neighbour distance. Supplying it
        replaces the simulation with a one-sample KS test, which suits
        only the case where edge effects are negligible or already
        corrected.
    nsim : int, default 199
        Number of CSR patterns simulated in ``window``. Ignored when
        ``cdf`` is given.
    seed : int, optional
        Seed for the simulation, for reproducible p-values.

    Returns
    -------
    RichResult
        keys: ``statistic`` (KS supremum distance), ``p_value``,
        ``nn_distances``, ``mean_nn``, ``n``, ``nsim``, ``method``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, sections 3.3-3.4.

    Diggle, P. J. (2003). *Statistical Analysis of Spatial Point
    Patterns*, 2nd edn. Arnold, London.
    """
    P = np.asarray(coords, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.ndim != 2:
        raise ValueError(f"coords must be (n, d); got shape {P.shape}.")
    n = P.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 events to form nearest-neighbour distances, got {n}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("coords must be finite.")

    nn = _nn_distances(P)

    if cdf is not None:
        ks = stats.kstest(nn, cdf)
        return RichResult(
            title="CSR test on nearest-neighbour distances",
            payload={
                "statistic": float(ks.statistic),
                "p_value": float(ks.pvalue),
                "nn_distances": nn,
                "mean_nn": float(nn.mean()),
                "n": int(n),
                "nsim": 0,
                "method": "One-sample KS against a supplied null CDF",
            },
        )

    nsim = int(nsim)
    if nsim < 1:
        raise ValueError(f"nsim must be at least 1, got {nsim}.")

    lo, hi = _as_window(window, P)
    rng = np.random.default_rng(seed)
    grid = np.sort(nn)

    sim_G = np.empty((nsim, grid.size))
    for i in range(nsim):
        sim_G[i] = _empirical_G(_nn_distances(rng.uniform(lo, hi, size=P.shape)), grid)

    G_bar = sim_G.mean(axis=0)
    d_obs = float(np.max(np.abs(_empirical_G(nn, grid) - G_bar)))

    # Each simulated pattern is scored against the mean of the others, so
    # none is compared with a reference it helped to build.
    d_sim = np.empty(nsim)
    for i in range(nsim):
        others = (G_bar * nsim - sim_G[i]) / (nsim - 1) if nsim > 1 else G_bar
        d_sim[i] = np.max(np.abs(sim_G[i] - others))

    p = (1.0 + float(np.sum(d_sim >= d_obs))) / (1.0 + nsim)

    return RichResult(
        title="CSR test on nearest-neighbour distances",
        payload={
            "statistic": d_obs,
            "p_value": p,
            "nn_distances": nn,
            "mean_nn": float(nn.mean()),
            "simulated_statistics": d_sim,
            "n": int(n),
            "nsim": nsim,
            "method": "Monte Carlo KS on G-hat against CSR simulated in the same window",
        },
    )


def cheatsheet():
    return "csrkstst: KS test of nearest-neighbour distance vs complete spatial randomness"
