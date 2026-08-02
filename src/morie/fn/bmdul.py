# morie.fn -- function file (rootcoder007/morie)
"""Bayesian multidimensional unfolding of rating data."""

from . import _array_core as np

from ._richresult import RichResult
from .procs import procrustes_rotation

__all__ = ["bayesian_mds_unfolding"]


def bayesian_mds_unfolding(ratings, n_dims=2, n_iter=3000, burnin=1000, seed=0, step=0.05):
    r"""Metropolis sampler for the unfolding model.

    .. math:: t_{ij} \sim N\big(\alpha - \|x_i - y_j\|^2,
              \sigma^2\big),

    a joint posterior over respondent ideal points X and stimulus
    locations Y -- the Bayesian version of the thermometer unfolding
    Armstrong's Ch. 4 develops, with the same rotation invariance
    handled by aligning kept draws of the *stimulus* configuration to
    its posterior-start position.

    Parameters
    ----------
    ratings : array-like, shape (n, q)
        Thermometer ratings (NaN = missing).
    n_dims : int, default 2
    n_iter, burnin, seed, step :
        Sampler controls.

    Returns
    -------
    RichResult
        keys: ``ideal_points`` (n, k posterior mean), ``stimuli``
        (q, k), ``alpha``, ``sigma``, ``acceptance``, ``n_kept``,
        ``n``, ``q``, ``method``.

    References
    ----------
    Bakker, R. & Poole, K. T. (2013). Bayesian metric
    multidimensional scaling. *Political Analysis*, 21(1), 125-140.
    (the sampling machinery; their bayesunfold extends it to exactly
    this model)

    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Ch. 4 (unfolding rating-scale
    data), p. 107.
    """
    T = np.asarray(ratings, dtype=float)
    if T.ndim != 2:
        raise ValueError("ratings must be 2-D (respondents x stimuli).")
    n, q = T.shape
    k = int(n_dims)
    if not 1 <= k <= min(n, q) - 1:
        raise ValueError(f"n_dims must lie in [1, {min(n, q) - 1}], got {k}.")
    n_iter, burnin = int(n_iter), int(burnin)
    if n_iter <= burnin:
        raise ValueError("n_iter must exceed burnin.")
    obs = ~np.isnan(T)
    if not obs.any():
        raise ValueError("all ratings are missing.")

    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.5, size=(n, k))
    Y = rng.normal(scale=0.5, size=(q, k))
    alpha = float(np.nanmax(T))
    sig2 = max(float(np.nanvar(T)), 1e-6)

    def pred(X, Y, alpha):
        diff = X[:, None, :] - Y[None, :, :]
        return alpha - (diff**2).sum(axis=2)

    def loglik(X, Y, alpha, sig2):
        r = (T - pred(X, Y, alpha))[obs]
        return -0.5 * (r**2).sum() / sig2 - 0.5 * obs.sum() * np.log(sig2)

    ll = loglik(X, Y, alpha, sig2)
    Y0 = None
    kept_X, kept_Y, accepted, proposals = [], [], 0, 0
    for it in range(n_iter):
        for M, rows in ((X, n), (Y, q)):
            for i in range(rows):
                old = M[i].copy()
                M[i] = old + rng.normal(scale=step, size=k)
                llp = loglik(X, Y, alpha, sig2)
                proposals += 1
                pen_new = -0.5 * (M[i] ** 2).sum() / 25.0
                pen_old = -0.5 * (old**2).sum() / 25.0
                if np.log(rng.random()) < (llp + pen_new) - (ll + pen_old):
                    ll = llp
                    accepted += 1
                else:
                    M[i] = old
        # alpha | rest (conjugate normal, flat prior)
        diff = X[:, None, :] - Y[None, :, :]
        sq = (diff**2).sum(axis=2)
        alpha = float(rng.normal((T + sq)[obs].mean(), np.sqrt(sig2 / obs.sum())))
        r = (T - pred(X, Y, alpha))[obs]
        sig2 = (1.0 + 0.5 * (r**2).sum()) / rng.gamma(1.0 + obs.sum() / 2.0)
        ll = loglik(X, Y, alpha, sig2)
        if it == burnin - 1:
            Y0 = Y.copy() - Y.mean(axis=0)
        if it >= burnin:
            Tm = procrustes_rotation(Y0, Y - Y.mean(axis=0))["rotation"]
            shift = Y.mean(axis=0)
            kept_Y.append((Y - shift) @ Tm)
            kept_X.append((X - shift) @ Tm)

    Xd, Yd = np.array(kept_X), np.array(kept_Y)
    return RichResult(
        payload={
            "ideal_points": Xd.mean(axis=0),
            "stimuli": Yd.mean(axis=0),
            "alpha": alpha,
            "sigma": float(np.sqrt(sig2)),
            "acceptance": accepted / max(proposals, 1),
            "n_kept": int(Xd.shape[0]),
            "n": int(n),
            "q": int(q),
            "method": "Bayesian unfolding: t ~ N(alpha - ||x - y||^2, s^2), MH over X and Y",
        }
    )


def cheatsheet():
    return "bmdul: joint MH over ideal points and stimuli; kept draws Procrustes-aligned"
