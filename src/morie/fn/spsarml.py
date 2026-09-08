# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood for spatial autoregressive models."""

from . import _array_core as np

from ._richresult import RichResult
from ._did import ols_fit

__all__ = ["schabenberger_sar_ml"]

MODELS = ("lag", "error")


def _concentrated(y, X, W, rho, model):
    """Profile beta and sigma^2 out at a given rho, returning -2 logL.

    Both models concentrate to a one-dimensional problem in the
    autoregressive parameter, which is why a scan plus refinement is
    reliable here where a general optimiser is not.
    """
    n = y.size
    A = np.eye(n) - rho * W
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0:
        return np.inf, None, np.nan
    if model == "lag":
        # Y = rho W Y + X beta + eps
        ystar = A @ y
        beta = ols_fit(X, ystar)
        r = ystar - X @ beta
    else:
        # Y = X beta + u, u = rho W u + eps: whiten both sides by A
        Xs, ys = A @ X, A @ y
        beta = ols_fit(Xs, ys)
        r = ys - Xs @ beta
    s2 = float(r @ r) / n
    if s2 <= 0:
        return np.inf, None, np.nan
    neg2 = n * np.log(2.0 * np.pi * s2) + n - 2.0 * logdetA
    return neg2, beta, s2


def schabenberger_sar_ml(x, y, w, model="error", n_grid=201):
    r"""Spatial autoregressive model by maximum likelihood.

    Two models share the name "SAR" in different literatures and they
    are NOT the same model. Both are provided and the choice is
    explicit rather than implied by the function name.

    ``model='error'`` is the simultaneous autoregressive model of
    Schabenberger and Gotway section 6.2.2.1, equations (6.36)-(6.37):

    .. math::
       Z = X\beta + e, \qquad e = \rho W e + \upsilon,

    which induces
    :math:`\Sigma_{SAR} = \sigma^2 (I-\rho W)^{-1}(I-\rho W')^{-1}`,
    equation (6.35). The autocorrelation lives entirely in the errors;
    :math:`\beta` keeps its ordinary interpretation.

    ``model='lag'`` regresses the response on its own spatial lag,
    :math:`Y = \rho W Y + X\beta + \varepsilon`. This is a different
    claim about the world -- an outcome at one site depends on the
    OUTCOME at neighbouring sites, so a change in :math:`x` at one site
    propagates through the whole system and :math:`\beta` is no longer
    a marginal effect.

    Both concentrate to one dimension. Profiling :math:`\beta` and
    :math:`\sigma^2` leaves

    .. math::
       -2\ln L(\rho) = n\ln(2\pi\hat\sigma^2(\rho)) + n
                       - 2\ln|I - \rho W|,

    and the Jacobian term :math:`\ln|I-\rho W|` is what least squares
    omits. That omission is not a small-sample matter: Whittle (1954)
    and Ord (1975) showed the least squares estimator of :math:`\rho`
    is INCONSISTENT, because :math:`Z` and :math:`\upsilon` are not
    independent. ``ols_rho`` is returned alongside for comparison, not
    as an alternative.

    The admissible range for :math:`\rho` is set by the eigenvalues of
    :math:`W`: :math:`1/\vartheta_{min} < \rho < 1/\vartheta_{max}`.
    For a row-standardised :math:`W`, :math:`\vartheta_{max} = 1`, so
    :math:`\rho < 1` while the lower bound may be below :math:`-1`.
    The search is confined to that interval and it is reported.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Design matrix. An intercept column is added if absent.
    y : array-like, shape (n,)
        Response.
    w : array-like, shape (n, n)
        Spatial proximity matrix, zero diagonal.
    model : {'error', 'lag'}
    n_grid : int
        Points in the initial scan over the admissible interval.

    Returns
    -------
    RichResult
        ``rho``, ``beta``, ``sigma2``, ``se`` (of beta), ``se_rho``,
        ``neg2loglik``, ``rho_bounds``, ``ols_rho``, ``row_standardised``.

    References
    ----------
    Schabenberger and Gotway (2005), section 6.2.2.1, equations
    (6.33)-(6.42), pp. 335-338. Whittle (1954), *Biometrika*
    41:434-449. Ord (1975), *JASA* 70:120-126.
    Cliff and Ord (1981). Anselin (1988) for the lag specification.

    Examples
    --------
    >>> import numpy as np
    >>> n = 40
    >>> W = np.zeros((n, n))
    >>> for i in range(n - 1):
    ...     W[i, i + 1] = W[i + 1, i] = 1.0
    >>> W = W / W.sum(axis=1, keepdims=True)
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([np.ones(n), rng.normal(size=n)])
    >>> e = np.linalg.solve(np.eye(n) - 0.5 * W, rng.normal(size=n) * 0.3)
    >>> out = schabenberger_sar_ml(X, X @ np.array([1.0, 2.0]) + e, W)
    >>> bool(-1.0 < out["rho"] < 1.0)
    True
    """
    if model not in MODELS:
        raise ValueError("model must be 'error' or 'lag', got %r." % model)
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yy = np.asarray(y, dtype=float).ravel()
    n = yy.size
    if X.shape[0] != n:
        X = X.T
    if X.shape[0] != n:
        raise ValueError("x has %d rows for %d responses." % (X.shape[0], n))
    if not np.any(np.all(np.isclose(X, 1.0), axis=0)):
        X = np.column_stack([np.ones(n), X])
    W = np.asarray(w, dtype=float)
    if W.shape != (n, n):
        raise ValueError(
            "w must be %d by %d, got %s." % (n, n, W.shape)
        )
    if np.any(np.abs(np.diag(W)) > 1e-12):
        raise ValueError(
            "w must have a zero diagonal; a site cannot be its own neighbour."
        )

    rs = bool(np.allclose(W.sum(axis=1), 1.0))
    ev = np.linalg.eigvals(W).real
    lo = 1.0 / ev.min() if ev.min() < 0 else -0.999
    hi = 1.0 / ev.max() if ev.max() > 0 else 0.999
    lo, hi = float(lo) + 1e-6, float(hi) - 1e-6

    grid = np.linspace(lo, hi, int(n_grid))
    vals = np.array([_concentrated(yy, X, W, r, model)[0] for r in grid])
    k = int(np.argmin(vals))
    a = grid[max(k - 1, 0)]
    b = grid[min(k + 1, grid.size - 1)]
    # golden-section refinement on the concentrated objective
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(200):
        if _concentrated(yy, X, W, c, model)[0] < _concentrated(
                yy, X, W, d, model)[0]:
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
        if abs(b - a) < 1e-12:
            break
    rho = float(0.5 * (a + b))
    neg2, beta, s2 = _concentrated(yy, X, W, rho, model)

    A = np.eye(n) - rho * W
    Xe = A @ X if model == "error" else X
    XtX = Xe.T @ Xe
    cov_beta = s2 * np.linalg.inv(XtX)
    # curvature of the concentrated objective gives a standard error
    # for rho; it ignores the covariance with beta and sigma^2, which
    # the full information matrix (6.42) would carry
    h = max(1e-5, 1e-4 * max(abs(rho), 1.0))
    f0 = neg2
    fp = _concentrated(yy, X, W, min(rho + h, hi), model)[0]
    fm = _concentrated(yy, X, W, max(rho - h, lo), model)[0]
    curv = (fp - 2.0 * f0 + fm) / h ** 2
    se_rho = float(np.sqrt(2.0 / curv)) if curv > 0 else float("nan")

    Wy = W @ yy
    ols_rho = float((yy @ W.T @ W @ yy) / (yy @ W.T @ W @ W @ yy)) \
        if abs(yy @ W.T @ W @ W @ yy) > 1e-12 else float("nan")

    return RichResult(
        payload={
            "estimate": np.atleast_1d(beta),
            "rho": rho,
            "beta": np.atleast_1d(beta),
            "sigma2": float(s2),
            "se": np.sqrt(np.diag(cov_beta)),
            "se_rho": se_rho,
            "beta_cov": cov_beta,
            "neg2loglik": float(neg2),
            "model": model,
            "model_note": (
                "the error model puts the dependence in the residuals and "
                "leaves beta a marginal effect; the lag model makes the "
                "outcome depend on neighbouring OUTCOMES, so a change at one "
                "site propagates and beta is no longer a marginal effect"
            ),
            "rho_bounds": (lo, hi),
            "row_standardised": rs,
            "bounds_note": (
                "admissible rho is 1/theta_min < rho < 1/theta_max from the "
                "eigenvalues of W; row-standardising fixes theta_max = 1 so "
                "rho < 1, but the lower bound can fall below -1"
            ),
            "ols_rho": ols_rho,
            "ols_note": (
                "the modified least-squares estimator of rho, shown for "
                "comparison only: ordinary least squares is INCONSISTENT "
                "here because Z and upsilon are not independent, which is "
                "the reason for maximum likelihood"
            ),
            "jacobian_note": (
                "the ln|I - rho W| term is exactly what least squares drops"
            ),
            "spatial_lag_mean": float(np.mean(Wy)),
            "n": n,
            "method": "Spatial autoregressive (%s) model by maximum "
                      "likelihood" % model,
        }
    )


def cheatsheet():
    return (
        "spsarml: SAR error (6.36) or spatial lag by concentrated ML, with "
        "the eigenvalue bounds on rho and why least squares is inconsistent"
    )
