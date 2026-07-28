# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic tightness / stochastic equicontinuity."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_weak_convergence_tightness"]


def kosorok_ch2_weak_convergence_tightness(X_n, rho=None, eps=0.1, delta_grid=None):
    r"""Asymptotic equicontinuity condition for tightness:

    .. math:: \lim_{\delta\downarrow 0} \limsup_n
              P^*\Big(\sup_{\rho(s,t)<\delta}
              |X_n(s) - X_n(t)| > \epsilon\Big) = 0.

    Given realisations of the process on a grid, this estimates the
    inner probability at a sequence of shrinking delta. Tightness
    requires the estimates to fall toward 0 as delta shrinks -- a
    process whose modulus of continuity does NOT vanish is not tight,
    and no amount of finite-dimensional convergence rescues it.

    Parameters
    ----------
    X_n : array-like, shape (n_rep, n_points)
        Replicated process paths on a common grid.
    rho : array-like, shape (n_points,), optional
        Index values; equally spaced on [0, 1] if omitted.
    eps : float, default 0.1
        Oscillation threshold.
    delta_grid : sequence of float, optional
        Shrinking neighbourhood widths.

    Returns
    -------
    RichResult
        keys: ``delta_grid``, ``probabilities``, ``decreasing``,
        ``eps``, ``n_rep``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (asymptotic tightness and equicontinuity).
    """
    P = np.atleast_2d(np.asarray(X_n, dtype=float))
    n_rep, n_pts = P.shape
    if n_pts < 3:
        raise ValueError("need at least 3 index points.")
    t = np.linspace(0, 1, n_pts) if rho is None else np.asarray(rho, dtype=float)
    if t.size != n_pts:
        raise ValueError("rho must match the number of index points.")
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if delta_grid is None:
        delta_grid = [0.5, 0.25, 0.1, 0.05]
    probs = []
    for d in delta_grid:
        d = float(d)
        if d <= 0:
            raise ValueError("delta values must be positive.")
        close = np.abs(t[:, None] - t[None, :]) < d
        iu = np.triu_indices(n_pts, 1)
        mask = close[iu]
        if not mask.any():
            probs.append(0.0)
            continue
        osc = np.abs(P[:, iu[0][mask]] - P[:, iu[1][mask]]).max(axis=1)
        probs.append(float(np.mean(osc > eps)))
    probs = np.array(probs)
    return RichResult(
        payload={"delta_grid": np.array([float(d) for d in delta_grid]),
                 "probabilities": probs,
                 "decreasing": bool(probs[-1] <= probs[0] + 1e-12),
                 "eps": eps, "n_rep": int(n_rep),
                 "method": "P*(modulus of continuity > eps) at shrinking delta"}
    )


def cheatsheet():
    return "ksr031: tightness = the oscillation probability must vanish with delta"
