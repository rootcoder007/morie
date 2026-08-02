"""Conditional autoregressive (CAR) model."""

import warnings

from . import _array_core as np

from ._containers import SpatialResult

__all__ = ["conditional_autoregressive"]


def _precision(W, D, rho, parameterization):
    if parameterization == "weighted":
        return D - rho * W
    return np.eye(W.shape[0]) - rho * W


def car_rho_bounds(W, parameterization="weighted"):
    r"""Valid interval for rho, from the eigenvalue condition.

    The joint distribution exists only where the precision matrix is
    positive definite. The book states the bound in terms of the
    eigenvalues :math:`\vartheta_i` of W (eq. 6.48):

    * ``identity``  -- :math:`Q = I - \rho W`, so
      :math:`\rho \in (1/\vartheta_{\min}, 1/\vartheta_{\max})`.
    * ``weighted`` -- :math:`Q = D - \rho W`, the same condition on the
      eigenvalues of :math:`D^{-1/2} W D^{-1/2}`.

    Returns ``(lo, hi)``, open interval.
    """
    W = np.asarray(W, dtype=float)
    if parameterization == "weighted":
        d = W.sum(axis=1)
        s = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
        M = (W * s[:, None]) * s[None, :]
    else:
        M = W
    ev = np.linalg.eigvalsh((M + M.T) / 2.0)
    lo = 1.0 / ev.min() if ev.min() < 0 else -np.inf
    hi = 1.0 / ev.max() if ev.max() > 0 else np.inf
    return float(lo), float(hi)


def car_rho_ols(Z, W, X=None):
    r"""Haining's least-squares estimator of rho.

    .. math::  \hat\rho_{OLS} = \frac{e'We}{e'W^2e}

    with e the OLS residual vector. The book notes that, unlike the SAR
    case, this estimator is CONSISTENT for a one-parameter CAR model
    (Haining 1990, p. 130), so it is a principled estimate in its own
    right and a good starting value for maximum likelihood.
    """
    Z = np.asarray(Z, dtype=float).ravel()
    W = np.asarray(W, dtype=float)
    X = np.ones((Z.size, 1)) if X is None else np.asarray(X, dtype=float)
    e = Z - X @ np.linalg.lstsq(X, Z, rcond=None)[0]
    denom = float(e @ (W @ (W @ e)))
    if abs(denom) < 1e-300:
        return 0.0
    return float(e @ (W @ e) / denom)


def conditional_autoregressive(Z, W, X=None, parameterization="weighted"):
    r"""Fit a Gaussian CAR model by maximum likelihood.

    The conditional specification (eqs. 6.43-6.44) is

    .. math::

        E[Z(s_i) \mid Z(s)_{-i}] = x(s_i)'\beta
            + \sum_j c_{ij}\,(Z(s_j) - x(s_j)'\beta),
        \qquad \mathrm{Var}[Z(s_i) \mid Z(s)_{-i}] = \sigma_i^2

    and by Hammersley-Clifford these generate a valid joint Gaussian with
    mean :math:`X\beta` and (eq. 6.45)

    .. math::  \Sigma_{CAR} = (I - C)^{-1}\Sigma_c

    Two one-parameter forms are supported, and they are NOT the same model:

    ``weighted`` (default)
        :math:`C = \rho D^{-1}W`, :math:`\Sigma_c = \sigma^2 D^{-1}`, giving
        precision :math:`(D - \rho W)/\sigma^2`. The conditional variance
        is :math:`\sigma^2/d_i` -- inversely proportional to the number of
        neighbours -- not constant. This is the Besag form and the one
        compatible with the ICAR limit.
    ``identity``
        :math:`C = \rho W`, :math:`\Sigma_c = \sigma^2 I`, giving precision
        :math:`(I - \rho W)/\sigma^2`. This is the case the book carries
        into estimation (eq. 6.47). Its valid rho interval is much
        narrower, roughly :math:`\pm 1/\vartheta_{\max}`.

    Estimation profiles the likelihood: for fixed :math:`\rho`,

    .. math::

        \hat\beta = (X'QX)^{-1}X'QZ, \qquad
        \hat\sigma^2 = \frac{1}{n}(Z - X\hat\beta)'Q(Z - X\hat\beta)

    .. math::

        \ell(\rho) = \tfrac{1}{2}\log|Q| - \tfrac{n}{2}\log\hat\sigma^2
                     - \tfrac{n}{2}

    which is then maximised over the VALID interval for :math:`\rho` --
    the open range where :math:`Q` is positive definite, from the
    eigenvalue condition of eq. (6.48). That interval includes zero and
    negative values: a CAR fit that cannot return :math:`\rho \le 0`
    cannot represent independence or competition, and will report spatial
    dependence in data that has none.

    Parameters
    ----------
    Z : array-like
        Response, shape ``(n,)``.
    W : array-like
        Symmetric adjacency weights, shape ``(n, n)``.
    X : array-like, optional
        Covariates. An intercept when omitted.
    parameterization : {'weighted', 'identity'}
        Which one-parameter form to fit (see above).

    Returns
    -------
    SpatialResult
        ``statistic`` is the ML estimate of :math:`\rho`. ``extra`` has
        ``beta``, ``sigma2``, ``tau2`` (an alias of ``sigma2``),
        ``loglik``, ``rho_ols`` (Haining's consistent estimator),
        ``rho_bounds`` and ``parameterization``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.2.2.2, eqs.
    (6.43)-(6.48), pp. 338-341. Besag (1974). Haining (1990), p. 130.
    """
    from ._sci_core import minimize_scalar

    Z = np.asarray(Z, dtype=float).ravel()
    W = np.asarray(W, dtype=float)
    n = Z.size
    if W.shape != (n, n):
        raise ValueError(f"`W` must be ({n}, {n}) to match `Z`")
    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError(
            "`W` must be symmetric: an asymmetric C gives a non-symmetric "
            "precision and no valid joint distribution (Hammersley-Clifford). "
            "Row-standardise via parameterization='weighted' instead of "
            "passing a row-standardised W."
        )
    if parameterization not in ("weighted", "identity"):
        raise ValueError("`parameterization` must be 'weighted' or 'identity'")
    X = np.ones((n, 1)) if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != n:
        raise ValueError("`X` must have one row per element of `Z`")

    D = np.diag(W.sum(axis=1))
    lo, hi = car_rho_bounds(W, parameterization)
    eps = 1e-6 * max(hi - lo, 1e-12)

    def neg_profile_ll(rho):
        Q = _precision(W, D, rho, parameterization)
        sign, logdet = np.linalg.slogdet(Q)
        if sign <= 0:
            return np.inf
        XtQX = X.T @ Q @ X
        try:
            beta = np.linalg.solve(XtQX, X.T @ Q @ Z)
        except np.linalg.LinAlgError:
            return np.inf
        r = Z - X @ beta
        s2 = float(r @ Q @ r) / n
        if s2 <= 0:
            return np.inf
        return -(0.5 * logdet - 0.5 * n * np.log(s2) - 0.5 * n)

    opt = minimize_scalar(neg_profile_ll, bounds=(lo + eps, hi - eps),
                          method="bounded",
                          options={"xatol": 1e-10 * max(hi - lo, 1.0)})
    rho = float(opt.x)
    if not np.isfinite(neg_profile_ll(rho)):
        warnings.warn(
            "CAR likelihood is not finite anywhere in the valid rho "
            "interval; falling back to rho = 0 (independence). The fit is "
            "OLS, not a spatial model.",
            stacklevel=2,
        )
        rho = 0.0

    Q = _precision(W, D, rho, parameterization)
    beta = np.linalg.solve(X.T @ Q @ X, X.T @ Q @ Z)
    r = Z - X @ beta
    s2 = float(r @ Q @ r) / n
    sign, logdet = np.linalg.slogdet(Q)
    loglik = (0.5 * logdet - 0.5 * n * np.log(s2) - 0.5 * n
              - 0.5 * n * np.log(2 * np.pi))

    return SpatialResult(
        name="conditional_autoregressive",
        statistic=rho,
        p_value=None,
        extra={"beta": beta, "sigma2": s2, "tau2": s2, "loglik": float(loglik),
               "rho_ols": car_rho_ols(Z, W, X), "rho_bounds": (lo, hi),
               "parameterization": parameterization},
    )


sgcar = conditional_autoregressive


def cheatsheet() -> str:
    return "conditional_autoregressive({}) -> Gaussian CAR by ML over the valid rho range."
