# morie.fn -- function file (rootcoder007/morie)
"""Bayesian metric MDS with posterior credible regions."""

from . import _array_core as np

from ._richresult import RichResult
from .mmdsf import metric_mds_torgerson
from .procs import procrustes_rotation

__all__ = ["bayesian_mds"]


def bayesian_mds(D_matrix, n_dims=2, n_iter=3000, burnin=1000, seed=0, step=0.05):
    r"""Random-walk Metropolis over the configuration.

    Bakker and Poole's move: treat the observed dissimilarities as
    noisy Euclidean distances,

    .. math:: d_{ij} \sim N\big(\|x_i - x_j\|, \sigma^2\big),

    put diffuse priors on the coordinates, and sample the
    configuration -- which is what turns a point-estimate MDS map into
    one with credible regions per point. Each kept draw is
    Procrustes-aligned to the Torgerson start before summarising,
    since the likelihood is rotation-invariant and raw draws would
    smear the posterior across orientations.

    Parameters
    ----------
    D_matrix : array-like, shape (n, n)
        Symmetric dissimilarities, zero diagonal.
    n_dims : int, default 2
    n_iter, burnin, seed :
        Sampler controls.
    step : float, default 0.05
        Random-walk proposal sd per coordinate.

    Returns
    -------
    RichResult
        keys: ``coordinates`` (posterior mean, aligned), ``ci_radius``
        (n, per-point 95% radius around the mean), ``sigma``,
        ``acceptance``, ``n_kept``, ``n``, ``method``.

    References
    ----------
    Bakker, R. & Poole, K. T. (2013). Bayesian metric
    multidimensional scaling. *Political Analysis*, 21(1), 125-140.
    """
    D = np.asarray(D_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D_matrix must be square.")
    n = D.shape[0]
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("D_matrix must be symmetric.")
    k = int(n_dims)
    if not 1 <= k <= n - 1:
        raise ValueError(f"n_dims must lie in [1, {n - 1}], got {k}.")
    n_iter, burnin = int(n_iter), int(burnin)
    if n_iter <= burnin:
        raise ValueError("n_iter must exceed burnin.")

    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(seed)

    X0 = metric_mds_torgerson(D, n_dims=k)["coordinates"]
    X = X0.copy()

    def dvec(X):
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff**2).sum(axis=2))[iu]

    dobs = D[iu]
    resid = dobs - dvec(X)
    sig2 = max(float((resid**2).mean()), 1e-6)

    def loglik(X, sig2):
        r = dobs - dvec(X)
        return -0.5 * (r**2).sum() / sig2 - 0.5 * dobs.size * np.log(sig2)

    ll = loglik(X, sig2)
    kept = []
    accepted = 0
    proposals = 0
    for it in range(n_iter):
        for i in range(n):  # one-point-at-a-time random walk
            prop = X.copy()
            prop[i] = X[i] + rng.normal(scale=step, size=k)
            llp = loglik(prop, sig2) - 0.5 * (prop**2).sum() / 100.0
            llc = ll - 0.5 * (X**2).sum() / 100.0
            proposals += 1
            if np.log(rng.random()) < llp - llc:
                X, ll = prop, loglik(prop, sig2)
                accepted += 1
        # sigma^2 | X (inverse-gamma with IG(1,1) prior)
        r = dobs - dvec(X)
        sig2 = (1.0 + 0.5 * (r**2).sum()) / rng.gamma(1.0 + dobs.size / 2.0)
        ll = loglik(X, sig2)
        if it >= burnin:
            aligned = procrustes_rotation(X0, X - X.mean(axis=0))["rotated"]
            kept.append(aligned)

    draws = np.array(kept)
    mean = draws.mean(axis=0)
    rad = np.percentile(np.sqrt(((draws - mean) ** 2).sum(axis=2)), 95, axis=0)

    return RichResult(
        payload={
            "coordinates": mean,
            "ci_radius": rad,
            "sigma": float(np.sqrt(sig2)),
            "acceptance": accepted / max(proposals, 1),
            "n_kept": int(draws.shape[0]),
            "n": int(n),
            "method": "Bayesian metric MDS (RW Metropolis, draws Procrustes-aligned)",
        }
    )


def cheatsheet():
    return "bymds: d_ij ~ N(||x_i - x_j||, s^2); MH over X, align draws before summarising"


# compact alias per ledger/NAMING.md
bayesianmds = bayesian_mds
