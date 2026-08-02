# morie.fn -- function file (rootcoder007/morie)
"""Kriging prediction error when covariance parameters are estimated."""

from . import _array_core as np

from ._richresult import RichResult
from ._schaben import MODELS
from ._schaben import variogram_model as _vgm

__all__ = ["schabenberger_kriging_pred_error"]


def _cov(D, model, nug, ps, rng):
    return (nug + ps) - _vgm(D, model, nug, ps, rng)


def schabenberger_kriging_pred_error(coords, z, target,
                                     variogram_model="exponential",
                                     nugget=None, psill=None, rng=None,
                                     jitter=0.05, n_jitter=24, seed=0):
    r"""Ordinary kriging with an honest prediction error, section 5.5.4.

    The plug-in kriging variance

    .. math::
       \hat\sigma^2_{ok}(s_0) = \hat C(0)
         - \sigma(\hat\theta)'\Sigma(\hat\theta)^{-1}\sigma(\hat\theta)
         + \frac{(1 - 1'\Sigma(\hat\theta)^{-1}\sigma(\hat\theta))^{2}}
                {1'\Sigma(\hat\theta)^{-1}1}

    is not the prediction error of the predictor actually being used.
    Substituting :math:`\hat\theta` into the predictor gives an
    ESTIMATE of the BLUP -- an EBLUP, which is no longer best -- while
    substituting into the variance formula gives the prediction error
    of a DIFFERENT predictor, the one that would apply had
    :math:`\theta` been known. The book is blunt that this "can be
    substantially biased", and always downward.

    Kackar and Harville (1984) supply the missing term. To first order,

    .. math::
       \mathrm{mse}[\hat\omega,\omega] \approx
         \mathrm{mse}[\tilde\omega,\omega]
         + \mathrm{tr}\{A(\theta)B(\theta)\},
       \quad
       A = \mathrm{Var}\!\left[\frac{\partial\tilde\omega}
                                    {\partial\theta}\right],
       \quad
       B = \mathrm{mse}[\hat\theta,\theta].

    Evaluating that at :math:`\hat\theta` removes only half the bias,
    so Prasad and Rao's approximately unbiased form, equation (5.53),
    doubles the correction:

    .. math::
       \mathrm{mse}[\tilde\omega,\omega,\hat\theta]
         + 2\,\mathrm{tr}\{A(\hat\theta)B(\hat\theta)\}.

    :math:`A` is obtained here by finite differences of the kriging
    weights with respect to the covariance parameters, and :math:`B`
    by perturbing the parameters over ``n_jitter`` draws of relative
    size ``jitter``. Both are approximations and are labelled as such:
    ``correction_share`` reports how much of the final variance the
    correction supplied, so an implausibly large one is visible rather
    than absorbed.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : array-like, shape (n,)
    target : array-like, shape (m, d) or (d,)
        Prediction locations.
    variogram_model : {'exponential', 'spherical', 'gaussian', 'linear'}
    nugget, psill, rng : float, optional
        Covariance parameters. Estimated by weighted least squares when
        omitted -- in which case they ARE estimated and the correction
        is the point of the exercise.
    jitter : float
        Relative perturbation used to approximate ``B``.
    n_jitter : int
        Number of perturbations.
    seed : int

    Returns
    -------
    RichResult
        ``prediction``, ``mse`` (corrected), ``mse_plugin``,
        ``correction``, ``correction_share``, ``se``, ``parameters``,
        ``parameters_estimated``.

    References
    ----------
    Schabenberger and Gotway (2005), section 5.5.4, equations
    (5.51)-(5.53), pp. 263-266. Kackar and Harville (1984).
    Harville and Jeske (1992). Prasad and Rao (1990).

    Examples
    --------
    >>> import numpy as np
    >>> rng_ = np.random.default_rng(0)
    >>> co = rng_.uniform(0, 10, size=(60, 2))
    >>> z = rng_.normal(size=60)
    >>> out = schabenberger_kriging_pred_error(co, z, [5.0, 5.0])
    >>> bool(out["mse"][0] >= out["mse_plugin"][0])
    True
    """
    model = variogram_model
    if model not in MODELS:
        raise ValueError("model must be one of %s, got %r." % (MODELS, model))
    P = np.atleast_2d(np.asarray(coords, dtype=float))
    zz = np.asarray(z, dtype=float).ravel()
    n = zz.size
    if P.shape[0] != n:
        P = P.T
    T = np.atleast_2d(np.asarray(target, dtype=float))
    if T.shape[1] != P.shape[1]:
        T = T.T
    if T.shape[1] != P.shape[1]:
        raise ValueError(
            "target has %d coordinate columns, coords has %d."
            % (T.shape[1], P.shape[1])
        )

    estimated = nugget is None or psill is None or rng is None
    if estimated:
        from ._schaben import fit_variogram_wls, matheron
        lag, gam, npair, _ = matheron(P, zz)
        f = fit_variogram_wls(lag, gam, npair, model)
        nugget, psill, rng = f["nugget"], f["psill"], f["range"]
    theta = np.array([float(nugget), float(psill), float(rng)])

    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    d0 = np.sqrt(np.sum((P[:, None, :] - T[None, :, :]) ** 2, axis=2))

    def _krige(t):
        C = _cov(D, model, *t) + np.eye(n) * 1e-10 * max(t[0] + t[1], 1e-12)
        c0 = _cov(d0, model, *t)
        Ci1 = np.linalg.solve(C, np.ones(n))
        Cic = np.linalg.solve(C, c0)
        denom = float(np.sum(Ci1))
        lam = Cic + np.outer(Ci1, (1.0 - np.sum(Cic, axis=0)) / denom)
        var = ((t[0] + t[1]) - np.sum(c0 * Cic, axis=0)
               + (1.0 - np.sum(Cic, axis=0)) ** 2 / denom)
        return lam, np.maximum(var, 0.0)

    lam, mse_plug = _krige(theta)
    pred = lam.T @ zz

    # A: variability of the kriging weights with respect to theta,
    # by central differences
    m = T.shape[0]
    dlam = np.zeros((3, n, m))
    for k in range(3):
        step = max(abs(theta[k]) * 1e-4, 1e-8)
        tp, tm = theta.copy(), theta.copy()
        tp[k] += step
        tm[k] = max(tm[k] - step, 1e-12)
        lp, _ = _krige(tp)
        lm, _ = _krige(tm)
        dlam[k] = (lp - lm) / (tp[k] - tm[k])

    # B: mse of theta-hat, approximated by perturbation
    gen = np.random.default_rng(int(seed))
    draws = theta[None, :] * (
        1.0 + float(jitter) * gen.normal(size=(int(n_jitter), 3))
    )
    draws = np.maximum(draws, 1e-10)
    Bmat = np.cov((draws - theta[None, :]).T, bias=True)
    Bmat = np.atleast_2d(Bmat)

    corr = np.zeros(m)
    Cth = _cov(D, model, *theta) + np.eye(n) * 1e-10 * max(theta[:2].sum(), 1e-12)
    for j in range(m):
        G = dlam[:, :, j]                      # (3, n)
        A = G @ Cth @ G.T                      # Var[d omega / d theta]
        corr[j] = float(np.trace(A @ Bmat))
    mse = mse_plug + 2.0 * corr                # equation (5.53)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(mse > 0, 2.0 * corr / mse, np.nan)
    return RichResult(
        payload={
            "estimate": pred,
            "prediction": pred,
            "mse": mse,
            "se": np.sqrt(np.maximum(mse, 0.0)),
            "mse_plugin": mse_plug,
            "correction": 2.0 * corr,
            "correction_share": share,
            "correction_note": (
                "Prasad-Rao equation (5.53): the Kackar-Harville term is "
                "DOUBLED, because evaluating it at theta-hat removes only "
                "half the bias"
            ),
            "plugin_note": (
                "mse_plugin is the prediction error of the predictor that "
                "would apply if theta were known -- not of the EBLUP "
                "actually being used, which is why it is biased downward"
            ),
            "parameters": {"nugget": float(theta[0]), "psill": float(theta[1]),
                           "range": float(theta[2])},
            "parameters_estimated": bool(estimated),
            "model": model,
            "n": n,
            "n_target": int(m),
            "method": "Ordinary kriging with Prasad-Rao corrected prediction "
                      "error",
        }
    )


def cheatsheet():
    return (
        "spkpe: kriging prediction error with the doubled Kackar-Harville "
        "correction (5.53), because the plug-in variance is the error of a "
        "different predictor"
    )
