# morie.fn -- function file (rootcoder007/morie)
"""Bayesian ideal-point estimation (Armstrong Ch 5)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bayesian_ideal_points", "bysid"]


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bayesian_ideal_points(x, n_iter: int = 400, burn: int = 100, seed: int = 0, deterministic_seed: int | None = None):
    """Bayesian ideal-point estimation, Metropolis-within-Gibbs surrogate
    for Clinton-Jackman-Rivers (2004).

    Prior: x_i ~ N(0, 1), alpha_j ~ N(0, 5), beta_j ~ N(0, 5).
    Likelihood: P(yea_ij) = sigmoid(alpha_j*(x_i - beta_j)).

    Parameters
    ----------
    x : (n, m) binary vote matrix, or 1-D vector (treated as single item).
    n_iter, burn : MCMC sweep lengths.
    seed : RNG seed.
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged: the user-supplied ``seed`` drives a fresh
        :class:`numpy.random.Generator`.

    Returns
    -------
    RichResult with keys: x_mean, x_sd, x_ci, alpha, beta, n_iter
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed

        rng = from_seed("bysid", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    M = np.asarray(x, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n, m = M.shape
    if n < 2:
        return RichResult(
            payload={
                "x_mean": np.full(n, np.nan),
                "x_sd": np.full(n, np.nan),
                "x_ci": np.full((n, 2), np.nan),
                "alpha": np.full(m, np.nan),
                "beta": np.full(m, np.nan),
                "n_iter": 0,
                "method": "bayesian_ideal_points",
            }
        )
    # Init from SVD
    Mc = np.nan_to_num(M - np.nanmean(M, axis=0, keepdims=True))
    try:
        u, s, vt = np.linalg.svd(Mc, full_matrices=False)
        x_cur = u[:, 0] * s[0]
    except np.linalg.LinAlgError:
        x_cur = rng.normal(size=n)
    x_cur = (x_cur - x_cur.mean()) / (x_cur.std() + 1e-12)
    a_cur = np.ones(m)
    b_cur = np.zeros(m)
    samples = []
    a_samples = []
    b_samples = []
    step_x = 0.4
    step_ab = 0.3

    mask = ~np.isnan(M)
    Msafe = np.where(mask, M, 0.0)

    def _cellwise_ll(xv, av, bv):
        """Per-cell log-likelihood, zero where the vote is missing."""
        Z = av[None, :] * (xv[:, None] - bv[None, :])
        P = _logistic(Z)
        return np.where(mask, Msafe * np.log(P + 1e-12) + (1 - Msafe) * np.log(1 - P + 1e-12), 0.0)

    def row_ll(xv, av, bv):
        return _cellwise_ll(xv, av, bv).sum(axis=1)  # (n,) -- x_i touches row i only

    def col_ll(xv, av, bv):
        return _cellwise_ll(xv, av, bv).sum(axis=0)  # (m,) -- a_j, b_j touch column j only

    # Metropolis-within-Gibbs: x_i enters only row i and a_j/b_j only column j, so
    # each coordinate is accepted or rejected on its own likelihood.  A single
    # joint accept/reject over all n coordinates has acceptance falling off
    # exponentially in n and leaves the chain pinned at its initial value.
    ll_x = row_ll(x_cur, a_cur, b_cur)
    acc_x = acc_a = acc_b = 0
    for t in range(n_iter):
        # Metropolis on x (per legislator)
        x_prop = x_cur + step_x * rng.normal(size=n)
        ll_prop = row_ll(x_prop, a_cur, b_cur)
        log_a = (ll_prop - 0.5 * x_prop**2) - (ll_x - 0.5 * x_cur**2)
        take = np.log(rng.uniform(size=n)) < log_a
        x_cur = np.where(take, x_prop, x_cur)
        ll_x = np.where(take, ll_prop, ll_x)
        acc_x += int(take.sum())

        # Metropolis on alpha (per item)
        ll_c = col_ll(x_cur, a_cur, b_cur)
        a_prop = a_cur + step_ab * rng.normal(size=m)
        ll_prop = col_ll(x_cur, a_prop, b_cur)
        log_a = (ll_prop - 0.5 * a_prop**2 / 25.0) - (ll_c - 0.5 * a_cur**2 / 25.0)
        take = np.log(rng.uniform(size=m)) < log_a
        a_cur = np.where(take, a_prop, a_cur)
        acc_a += int(take.sum())

        # Metropolis on beta (per item)
        ll_c = col_ll(x_cur, a_cur, b_cur)
        b_prop = b_cur + step_ab * rng.normal(size=m)
        ll_prop = col_ll(x_cur, a_cur, b_prop)
        log_a = (ll_prop - 0.5 * b_prop**2 / 25.0) - (ll_c - 0.5 * b_cur**2 / 25.0)
        take = np.log(rng.uniform(size=m)) < log_a
        b_cur = np.where(take, b_prop, b_cur)
        acc_b += int(take.sum())

        ll_x = row_ll(x_cur, a_cur, b_cur)

        if t < burn:
            # Robbins-Monro step tuning towards the 0.44 optimum for scalar moves.
            step_x *= np.exp((acc_x / ((t + 1) * n) - 0.44) * 0.5)
            step_ab *= np.exp((0.5 * (acc_a + acc_b) / ((t + 1) * m) - 0.44) * 0.5)
            step_x = float(np.clip(step_x, 1e-3, 5.0))
            step_ab = float(np.clip(step_ab, 1e-3, 5.0))

        if t >= burn:
            # Renorm for identification
            xs = (x_cur - x_cur.mean()) / (x_cur.std() + 1e-12)
            samples.append(xs.copy())
            a_samples.append(a_cur.copy())
            b_samples.append(b_cur.copy())
    arr = np.array(samples) if samples else np.zeros((1, n))
    x_mean = arr.mean(axis=0)
    x_sd = arr.std(axis=0)
    x_ci = np.percentile(arr, [2.5, 97.5], axis=0).T
    a_mean = np.mean(a_samples, axis=0) if a_samples else np.full(m, np.nan)
    b_mean = np.mean(b_samples, axis=0) if b_samples else np.full(m, np.nan)
    return RichResult(
        title="Bayesian ideal points (Metropolis-within-Gibbs)",
        summary_lines=[("posterior draws", len(samples)), ("n legislators", n), ("m items", m)],
        payload={
            "x_mean": x_mean,
            "x_sd": x_sd,
            "x_ci": x_ci,
            "alpha": a_mean,
            "beta": b_mean,
            "n_iter": int(n_iter),
            "method": "bayesian_ideal_points",
        },
    )


bysid = bayesian_ideal_points


def cheatsheet():
    return "bysid: Bayesian ideal points (CJR Metropolis-within-Gibbs)."


# CANONICAL TEST
# >>> rng = np.random.default_rng(1)
# >>> X = rng.normal(size=15)
# >>> Y = (X[:, None] + rng.normal(size=(15, 8)) > 0).astype(int)
# >>> r = bayesian_ideal_points(Y, n_iter=200, burn=50)
# >>> assert abs(np.corrcoef(r["x_mean"], X)[0,1]) > 0.4
