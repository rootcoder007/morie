# morie.fn -- function file (rootcoder007/morie)
"""Complete spatial randomness test via Monte Carlo envelopes on K(r)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["csr_test"]


def _ripley_k(P, radii, area):
    r"""Ripley's K without edge correction.

    .. math:: \hat K(r) = \frac{|A|}{n^2}\sum_i \sum_{j \ne i} 1(d_{ij} \le r)

    Edge effects are left in deliberately: they are shared with the
    simulated patterns, which are drawn in the same window, so they
    cancel in the envelope comparison.
    """
    n = P.shape[0]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    counts = (d[None, :, :] <= radii[:, None, None]).sum(axis=(1, 2))
    return area * counts / (n * n)


def _as_window(window, P):
    """Normalise ``window`` to (lo, hi); d rows of (min, max) or flat."""
    d = P.shape[1]
    if window is None:
        return P.min(axis=0), P.max(axis=0)
    W = np.asarray(window, dtype=float)
    if W.size != 2 * d:
        raise ValueError(f"window must give {d} (min, max) pairs for {d}-dimensional coords; got shape {W.shape}.")
    pair = W.reshape(d, 2)
    lo, hi = pair[:, 0], pair[:, 1]
    if np.any(hi <= lo):
        raise ValueError("window upper bounds must exceed lower bounds.")
    return lo, hi


def csr_test(coords, window=None, nsim=99, cdf=None, radii=None, seed=None):
    r"""Test a point pattern against CSR using Ripley's K.

    Compares the observed :math:`\hat K(r)` with pointwise Monte Carlo
    envelopes built from patterns simulated under complete spatial
    randomness in the same window. Under CSR in the plane
    :math:`K(r) = \pi r^2`, but that identity holds only on the infinite
    plane; inside a bounded window the uncorrected estimator is biased
    downward, so the envelopes -- not the closed form -- are the right
    reference. Simulating in the same window makes the bias common to
    both sides, so it cancels.

    This is the second-order counterpart of
    :func:`morie.fn.csrkstst.kstest_csr`, which tests the same null
    through first-order nearest-neighbour distances. K looks at all
    inter-point distances up to :math:`r` rather than the nearest one, so
    it can see clustering at one scale and regularity at another; the
    nearest-neighbour test cannot.

    The reported p-value comes from the maximum absolute deviation of
    :math:`\hat K` from the simulated mean, ranked among the simulations,

    .. math:: p = \frac{1 + \#\{u^{(b)} \ge u^{obs}\}}{1 + n_{sim}}

    a single number over the whole radius range. Pointwise envelopes are
    returned as well, but reading significance off the radius where the
    curve happens to leave them would be multiple testing.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Event locations.
    window : array-like, optional
        Observation region as ``d`` (min, max) pairs. Defaults to the
        bounding box of ``coords``.
    nsim : int, default 99
        Number of CSR patterns simulated.
    cdf : callable, optional
        Null CDF of the deviation statistic, replacing the simulation.
    radii : array-like, optional
        Radii at which K is evaluated. Defaults to 20 points from 0 to a
        quarter of the smallest window side, the usual guidance being to
        keep r well below the window size.
    seed : int, optional
        Seed for the simulation.

    Returns
    -------
    RichResult
        keys: ``statistic`` (max deviation), ``p_value``, ``radii``,
        ``k_observed``, ``k_mean``, ``k_lower``, ``k_upper``, ``n``,
        ``nsim``, ``area``, ``method``.

    References
    ----------
    Ripley, B. D. (1977). Modelling spatial patterns. *Journal of the
    Royal Statistical Society, Series B*, 39(2), 172-212.

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, section 3.4.
    """
    P = np.asarray(coords, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.ndim != 2:
        raise ValueError(f"coords must be (n, d); got shape {P.shape}.")
    n = P.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 events, got {n}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("coords must be finite.")

    lo, hi = _as_window(window, P)
    side = hi - lo
    area = float(np.prod(side))

    if radii is None:
        radii = np.linspace(0.0, float(side.min()) / 4.0, 20)[1:]
    r = np.asarray(radii, dtype=float).ravel()
    if r.size == 0 or np.any(r <= 0):
        raise ValueError("radii must be positive and non-empty.")

    k_obs = _ripley_k(P, r, area)

    if cdf is not None:
        stat = float(np.max(np.abs(k_obs - np.pi * r**2)))
        return RichResult(
            title="CSR test on Ripley's K",
            payload={
                "statistic": stat,
                "p_value": float(1.0 - cdf(stat)),
                "radii": r,
                "k_observed": k_obs,
                "n": int(n),
                "nsim": 0,
                "area": area,
                "method": "Ripley K deviation from pi r^2 against a supplied null CDF",
            },
        )

    nsim = int(nsim)
    if nsim < 1:
        raise ValueError(f"nsim must be at least 1, got {nsim}.")
    rng = np.random.default_rng(seed)
    sims = np.empty((nsim, r.size))
    for i in range(nsim):
        sims[i] = _ripley_k(rng.uniform(lo, hi, size=P.shape), r, area)

    k_mean = sims.mean(axis=0)
    u_obs = float(np.max(np.abs(k_obs - k_mean)))
    # Score each simulation against the mean of the others, so none is
    # compared with a reference it helped to build.
    u_sim = np.empty(nsim)
    for i in range(nsim):
        others = (k_mean * nsim - sims[i]) / (nsim - 1) if nsim > 1 else k_mean
        u_sim[i] = np.max(np.abs(sims[i] - others))
    p = (1.0 + float(np.sum(u_sim >= u_obs))) / (1.0 + nsim)

    return RichResult(
        title="CSR test on Ripley's K",
        payload={
            "statistic": u_obs,
            "p_value": p,
            "radii": r,
            "k_observed": k_obs,
            "k_mean": k_mean,
            "k_lower": sims.min(axis=0),
            "k_upper": sims.max(axis=0),
            "n": int(n),
            "nsim": nsim,
            "area": area,
            "method": "Monte Carlo envelope on Ripley's K (Ripley 1977)",
        },
    )


def cheatsheet():
    return "mrkcsr: CSR test via Monte Carlo envelopes on Ripley's K"


# compact alias per ledger/NAMING.md
csrtest = csr_test
