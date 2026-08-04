# morie.fn -- function file (rootcoder007/morie)
"""Moment-inequality criterion and confidence set."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bound_nonlinear"]


def bound_nonlinear(data, g, theta_grid, alpha=0.05, B=500, seed=0):
    r"""Criterion-function inference for a model defined by moment
    INEQUALITIES :math:`E[g_j(X, \theta)] \le 0`, after Chernozhukov,
    Hong and Tamer (2007).

    The sample criterion is

    .. math:: Q_n(\theta) = \sum_j \left[\frac{\sqrt n\,\bar
              g_j(\theta)}{\hat\sigma_j(\theta)}\right]_+^2 ,

    with :math:`[x]_+ = \max(x, 0)`: only VIOLATED inequalities
    contribute. On the identified set
    :math:`\Theta_I = \{\theta : E[g_j] \le 0\ \forall j\}` the
    population criterion is exactly zero, and off it, positive --
    which is what makes the set estimable as a level set of
    :math:`Q_n`.

    The confidence region is the level set
    :math:`\{\theta : Q_n(\theta) \le \hat c_{1-\alpha}\}`, with the
    critical value taken from a multiplier bootstrap of the RECENTRED
    moments -- recentred because the null distribution is driven by
    the binding inequalities at their boundary, not by however far
    the sample moments happen to sit inside. Skipping the recentring
    makes the critical value grow with slack and the region
    conservative to the point of uselessness; the tests check the
    region rather than the internals.

    Two facts the construction guarantees and the tests assert: the
    region is a superset of the criterion's argmin (the set estimate),
    and a parameter deep inside the identified set has
    :math:`Q_n = 0` exactly since every sample moment is negative.

    Parameters
    ----------
    data : array-like, shape (n, ...)
        Observations, passed row-wise to ``g``.
    g : callable
        ``g(data, theta)`` returning an (n, J) array of moment
        evaluations, one column per inequality, oriented so that the
        model says :math:`E[g_j] \le 0`.
    theta_grid : array-like
        Candidate parameter values; scalars or vectors, one per row.
    alpha : float, default 0.05
        Miss probability for the confidence region.
    B : int, default 500
        Multiplier-bootstrap draws.
    seed : int, default 0
        Bootstrap seed.

    Returns
    -------
    RichResult
        keys: ``theta_grid``, ``criterion``, ``critical_value``,
        ``in_confidence_set``, ``set_estimate`` (argmin of the
        criterion), ``confidence_set_bounds``, ``n_binding_max``,
        ``n``, ``J``, ``method``.

    References
    ----------
    Chernozhukov, V., Hong, H. and Tamer, E. (2007), "Estimation and
    confidence regions for parameter sets in econometric models",
    *Econometrica* 75:1243-1284. Andrews, D. W. K. and Soares, G.
    (2010), *Econometrica* 78:119-157, for moment selection.
    """
    d = np.asarray(data, dtype=float)
    n = d.shape[0]
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    grid = np.atleast_1d(np.asarray(theta_grid, dtype=float))
    if grid.ndim == 1:
        thetas = list(grid)
    else:
        thetas = [grid[i] for i in range(grid.shape[0])]
    a = float(alpha)
    if not 0 < a < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {a}.")
    rng = np.random.default_rng(seed)

    Q = np.empty(len(thetas))
    crit = np.empty(len(thetas))
    nbind = 0
    for i, th in enumerate(thetas):
        G = np.atleast_2d(np.asarray(g(d, th), dtype=float))
        if G.shape[0] != n:
            G = G.T
        if G.shape[0] != n:
            raise ValueError("g must return one row per observation.")
        gbar = G.mean(axis=0)
        sd = G.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        t = np.sqrt(n) * gbar / sd
        Q[i] = float(np.sum(np.maximum(t, 0.0) ** 2))
        nbind = max(nbind, int(np.sum(np.abs(t) < 2.0)))
        # multiplier bootstrap of the RECENTRED moments: the null is
        # driven by inequalities AT their boundary, so the centred
        # process is what the critical value must come from
        Z = (G - gbar) / sd
        mult = rng.standard_normal((int(B), n))
        boot_t = mult @ Z / np.sqrt(n)
        # only the (nearly) binding moments can contribute under the
        # null; slack ones are pushed to -inf by the sqrt(n) scaling
        binding = t > -np.sqrt(2 * np.log(np.log(max(n, 3))))
        bq = np.sum(np.maximum(boot_t[:, binding], 0.0) ** 2, axis=1) \
            if binding.any() else np.zeros(int(B))
        crit[i] = float(np.quantile(bq, 1 - a))
    inset = Q <= crit
    argmin = np.flatnonzero(Q <= Q.min() + 1e-12)
    flat = grid if grid.ndim == 1 else np.arange(len(thetas))
    cs = (float(np.min(flat[inset])), float(np.max(flat[inset]))) \
        if inset.any() and grid.ndim == 1 else None
    return RichResult(payload={
        "theta_grid": grid, "criterion": Q, "critical_value": crit,
        "in_confidence_set": inset,
        "set_estimate": flat[argmin] if grid.ndim == 1 else argmin,
        "confidence_set_bounds": cs,
        "positive_part_note": "only VIOLATED inequalities enter Q_n; deep "
                              "inside the identified set every sample moment "
                              "is negative and Q_n is exactly zero",
        "recentring_note": "the bootstrap recentres the moments because the "
                           "null sits at the boundary of the binding "
                           "inequalities, not at the sample slack",
        "n_binding_max": nbind,
        "n": int(n), "J": int(np.atleast_2d(np.asarray(
            g(d, thetas[0]), dtype=float)).shape[-1]),
        "method": "Chernozhukov-Hong-Tamer criterion-function confidence "
                  "region for moment inequalities"})


def cheatsheet():
    return "bndnln: [x]_+ in the criterion, recentred bootstrap for the cutoff"


# compact alias per ledger/NAMING.md
boundnonlinear = bound_nonlinear
