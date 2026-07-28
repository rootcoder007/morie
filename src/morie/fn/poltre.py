# morie.fn -- function file (rootcoder007/morie)
"""Polya tree prior for density estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["polya_tree_density"]


def polya_tree_density(x, at=None, depth=8, c=1.0, base="normal",
                       base_params=None):
    r"""Nonparametric density under a Polya tree prior.

    The unit interval is split recursively; at level :math:`j` each
    node's mass splits by a Beta draw whose parameters grow with depth,

    .. math:: \alpha_{j} = c\,j^{2},

    and the posterior updates each Beta conjugately with the counts
    falling in its two children.

    The :math:`j^2` growth is not arbitrary and it is the crux of the
    construction. Ferguson's condition for the resulting random measure
    to be ABSOLUTELY CONTINUOUS with probability one is that
    :math:`\sum_j \alpha_j^{-1} < \infty`, which :math:`j^2` satisfies
    and :math:`j` does not. Set :math:`\alpha_j = cj` and the prior
    puts all its mass on discrete distributions -- the posterior still
    computes, and the "density" it returns is meaningless.
    ``absolutely_continuous`` records which regime the chosen growth
    puts you in.

    Unlike a Dirichlet process, a Polya tree can therefore model a
    density directly rather than requiring a mixture. What it gives up
    is smoothness across partition boundaries: the fitted density is
    discontinuous at every dyadic cut, and those artefacts are a
    property of the PARTITION rather than of the data.
    ``partition_artefacts`` counts the cuts at the finest level.

    Parameters
    ----------
    x : array-like, shape (n,)
    at : array-like, optional
    depth : int
        Levels of recursive splitting.
    c : float
        Prior strength.
    base : {'normal', 'uniform'}
        Centring measure.
    base_params : tuple, optional

    Returns
    -------
    RichResult
        ``density``, ``at``, ``absolutely_continuous``,
        ``partition_artefacts``, ``integral``, ``effective_bins``.

    References
    ----------
    Mauldin, Sudderth and Williams (1992), *Annals of Statistics*
    20:1203-1221.
    Lavine (1992), *Annals of Statistics* 20:1222-1235.
    Ferguson (1974) for the absolute-continuity condition.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> out = polya_tree_density(rng.normal(size=300), depth=6)
    >>> bool(out["absolutely_continuous"])
    True
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 2:
        raise ValueError("need at least 2 observations, got %d." % n)
    if np.any(~np.isfinite(v)):
        raise ValueError("x contains non-finite values.")
    J = int(depth)
    if J < 1:
        raise ValueError("depth must be at least 1.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if base not in ("normal", "uniform"):
        raise ValueError("base must be 'normal' or 'uniform'.")

    if base == "normal":
        mu, sd = (float(np.mean(v)), float(np.std(v, ddof=1)) or 1.0) \
            if base_params is None else base_params
        import math
        def cdf(t):
            return 0.5 * np.array([math.erfc(-(ti - mu) / (sd * math.sqrt(2)))
                                   for ti in np.atleast_1d(t)])
        def icdf(u):
            out = np.empty_like(u)
            for i, q in enumerate(np.atleast_1d(u)):
                lo_, hi_ = mu - 12 * sd, mu + 12 * sd
                for _ in range(80):
                    mid = 0.5 * (lo_ + hi_)
                    if float(cdf(mid)[0]) < q:
                        lo_ = mid
                    else:
                        hi_ = mid
                out[i] = 0.5 * (lo_ + hi_)
            return out
        dens_base = lambda t: np.exp(-0.5 * ((t - mu) / sd) ** 2) \
            / (sd * np.sqrt(2 * np.pi))
    else:
        a, b = (float(v.min()), float(v.max())) if base_params is None \
            else base_params
        if b <= a:
            raise ValueError("uniform base needs b > a.")
        cdf = lambda t: np.clip((np.atleast_1d(t) - a) / (b - a), 0, 1)
        icdf = lambda u: a + np.atleast_1d(u) * (b - a)
        dens_base = lambda t: np.full_like(np.atleast_1d(t), 1.0 / (b - a))

    u = np.clip(cdf(v), 1e-12, 1 - 1e-12)
    # posterior mass multiplier on each dyadic cell at the finest level
    mult = np.ones(2 ** J)
    for j in range(1, J + 1):
        alpha = c * j ** 2
        cells = 2 ** j
        edges = np.linspace(0.0, 1.0, cells + 1)
        idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, cells - 1)
        counts = np.bincount(idx, minlength=cells).astype(float)
        for p in range(cells // 2):
            nl, nr = counts[2 * p], counts[2 * p + 1]
            tot = alpha * 2 + nl + nr
            wl = (alpha + nl) / tot * 2.0
            wr = (alpha + nr) / tot * 2.0
            rep = 2 ** (J - j)
            mult[(2 * p) * rep:(2 * p + 1) * rep] *= wl
            mult[(2 * p + 1) * rep:(2 * p + 2) * rep] *= wr

    grid = (np.linspace(v.min() - 0.5, v.max() + 0.5, 400)
            if at is None else np.asarray(at, dtype=float).ravel())
    ug = np.clip(cdf(grid), 1e-12, 1 - 1e-12)
    cell = np.clip((ug * 2 ** J).astype(int), 0, 2 ** J - 1)
    dens = dens_base(grid) * mult[cell]
    integral = float(np.trapezoid(dens, grid)) if grid.size > 1 else np.nan
    growth_ok = True                        # alpha_j = c j^2 by construction
    return RichResult(
        payload={
            "estimate": dens,
            "density": dens,
            "at": grid,
            "cell_multipliers": mult,
            "absolutely_continuous": growth_ok,
            "continuity_note": (
                "alpha_j = c j^2 makes sum 1/alpha_j finite, which is "
                "Ferguson's condition for the random measure to be "
                "absolutely continuous; alpha_j = c j fails it and the prior "
                "then puts all its mass on discrete distributions while the "
                "posterior still computes"
            ),
            "partition_artefacts": int(2 ** J - 1),
            "artefact_note": (
                "the fitted density is discontinuous at every dyadic cut; "
                "those jumps belong to the partition, not to the data"
            ),
            "integral": integral,
            "effective_bins": float(np.sum(mult > 0)),
            "depth": J,
            "c": float(c),
            "base": base,
            "n": int(n),
            "method": "Polya tree density estimate",
        }
    )


def cheatsheet():
    return (
        "poltre: Polya tree density with Ferguson's absolute-continuity "
        "condition enforced and partition artefacts named"
    )
