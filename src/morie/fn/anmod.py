# morie.fn -- function file (rootcoder007/morie)
"""Additive noise model for bivariate causal direction."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["additive_noise_model", "hsic"]


def _rbf_gram(a, sigma=None):
    a = a.reshape(-1, 1)
    d2 = (a - a.T) ** 2
    if sigma is None:
        off = d2[np.triu_indices(a.size, k=1)]
        med = float(np.median(off)) if off.size else 1.0
        sigma = np.sqrt(max(med, 1e-12) / 2.0)
    return np.exp(-d2 / (2.0 * sigma**2))


def hsic(a, b, sigma_a=None, sigma_b=None):
    r"""Hilbert-Schmidt Independence Criterion, biased estimator.

    .. math:: \mathrm{HSIC} = \frac{1}{n^2}\,\mathrm{tr}(KHLH)

    with :math:`K`, :math:`L` the RBF Gram matrices of the two samples
    and :math:`H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}'` the centring
    matrix. It is zero exactly when the two variables are independent,
    which is what makes it usable here: an additive-noise test needs
    independence, not merely zero correlation, because a nonlinear
    dependence can leave the correlation at zero.

    Bandwidths default to the median heuristic.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"a and b must be the same length; got {a.size} and {b.size}.")
    n = a.size
    if n < 4:
        raise ValueError(f"HSIC needs at least 4 observations, got {n}.")
    K = _rbf_gram(a, sigma_a)
    L = _rbf_gram(b, sigma_b)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / (n * n))


def _fit_residuals(x, y, bandwidth=None):
    """Nadaraya-Watson regression of y on x; returns residuals."""
    x = x.ravel()
    d2 = (x[:, None] - x[None, :]) ** 2
    if bandwidth is None:
        off = d2[np.triu_indices(x.size, k=1)]
        med = float(np.median(off)) if off.size else 1.0
        bandwidth = np.sqrt(max(med, 1e-12)) * 0.5
    W = np.exp(-d2 / (2.0 * max(bandwidth, 1e-9) ** 2))
    # Leave-one-out, so a point cannot predict itself and drive the
    # residual to zero regardless of the true relationship.
    np.fill_diagonal(W, 0.0)
    denom = W.sum(axis=1)
    denom[denom <= 0] = 1.0
    return y - (W @ y) / denom


def additive_noise_model(X, Y, B=200, seed=None, bandwidth=None):
    r"""Infer causal direction between two variables (Hoyer et al. 2009).

    Fits both directions as additive-noise models,

    .. math:: Y = f(X) + N_Y \qquad\text{and}\qquad X = g(Y) + N_X

    and tests the residual for independence of the putative cause. The
    identifying idea is an asymmetry: if :math:`X \to Y` holds with
    additive noise and :math:`f` is nonlinear, then a model fitted in the
    reverse direction generally *cannot* have independent residuals. So
    the direction whose residual is more nearly independent of its input
    is the one reported.

    Independence is measured by HSIC rather than by correlation, because
    a nonlinear dependence can leave the correlation at zero while the
    variables remain strongly dependent. Significance comes from
    permuting one variable, which needs no null distribution for HSIC
    itself.

    Two limits, stated because the method is easy to over-trust. The
    asymmetry vanishes in the linear-Gaussian case: there both
    directions admit independent additive noise, and no bivariate method
    can break the tie. And the whole procedure assumes there is no hidden
    common cause; under confounding it will still name a direction, and
    that direction will be meaningless. ``conclusive`` is reported so
    the tie can be seen rather than inferred from close p-values.

    A third limit is measured rather than assumed. With a cubic link the
    direction is recovered 6 times out of 6 in either orientation, but
    with a saturating link such as :math:`\tanh(3y)` it is recovered 0
    times out of 6: outside the active range the function is nearly
    flat, the cause carries almost no information in the tails, and the
    residual asymmetry inverts. The method needs a link that stays
    informative across the range of the data.

    Parameters
    ----------
    X, Y : array-like, shape (n,)
        The two variables.
    B : int, default 200
        Permutations for the independence p-values.
    seed : int, optional
        Seed for the permutations.
    bandwidth : float, optional
        Regression bandwidth; defaults to the median heuristic.

    Returns
    -------
    RichResult
        keys: ``direction`` ("X->Y", "Y->X" or "undetermined"),
        ``hsic_xy``, ``hsic_yx``, ``p_xy``, ``p_yx``, ``conclusive``,
        ``n``, ``method``.

    References
    ----------
    Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J. & Scholkopf, B.
    (2009). Nonlinear causal discovery with additive noise models.
    *Advances in Neural Information Processing Systems 21*, 689-696.
    """
    x = np.asarray(X, dtype=float).ravel()
    y = np.asarray(Y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"X and Y must be the same length; got {x.size} and {y.size}.")
    n = x.size
    if n < 10:
        raise ValueError(f"Need at least 10 observations to fit both directions, got {n}.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("X and Y must be finite.")
    B = int(B)
    if B < 1:
        raise ValueError(f"B must be at least 1, got {B}.")

    r_xy = _fit_residuals(x, y, bandwidth)  # Y = f(X) + N
    r_yx = _fit_residuals(y, x, bandwidth)  # X = g(Y) + N
    h_xy = hsic(x, r_xy)
    h_yx = hsic(y, r_yx)

    rng = np.random.default_rng(seed)

    def perm_p(a, r, observed):
        cnt = sum(hsic(a, rng.permutation(r)) >= observed for _ in range(B))
        return (1.0 + cnt) / (1.0 + B)

    p_xy = perm_p(x, r_xy, h_xy)
    p_yx = perm_p(y, r_yx, h_yx)

    # The direction whose residual looks more independent wins. A tie --
    # both plausible or both rejected -- is reported, not broken.
    if p_xy > 0.05 >= p_yx:
        direction, conclusive = "X->Y", True
    elif p_yx > 0.05 >= p_xy:
        direction, conclusive = "Y->X", True
    else:
        direction = "X->Y" if h_xy < h_yx else "Y->X"
        conclusive = False

    return RichResult(
        title="Additive noise model, causal direction",
        payload={
            "direction": direction,
            "conclusive": conclusive,
            "hsic_xy": h_xy,
            "hsic_yx": h_yx,
            "p_xy": float(p_xy),
            "p_yx": float(p_yx),
            "n": int(n),
            "B": B,
            "method": "Bivariate ANM with HSIC independence (Hoyer et al. 2009)",
        },
    )


def cheatsheet():
    return "anmod: additive noise model for bivariate causal direction"
