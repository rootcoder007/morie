# morie.fn -- function file (rootcoder007/morie)
"""Maximum and restricted maximum likelihood variogram estimation."""

from . import _array_core as np

from ._richresult import RichResult
from ._schaben import (MODELS, _nelder_mead, gaussian_neg2loglik,
                       reml_neg2loglik)

__all__ = ["schabenberger_ml_variogram"]


def schabenberger_ml_variogram(coords, z, variogram_model="exponential",
                               method="ml", X=None):
    r"""Covariance parameters by ML or REML, Schabenberger eqs (4.35)/(4.39).

    Maximum likelihood minimises

    .. math::
       \varphi(\mu;\theta) = \ln|\Sigma(\theta)| + n\ln 2\pi
         + (Z - X\beta)'\Sigma(\theta)^{-1}(Z - X\beta),

    with the mean profiled out by generalised least squares (4.36).
    Restricted maximum likelihood instead maximises the likelihood of
    error contrasts :math:`KZ` chosen so that :math:`E[KZ] = 0`, which
    adds the term :math:`\ln|X'\Sigma^{-1}X|`.

    That extra term is the entire difference, and it matters. ML makes
    no allowance for the degrees of freedom spent estimating the mean,
    so its variance estimates are biased downward -- for independent
    observations with unknown mean the bias is exactly
    :math:`-\theta/n` (p. 167). REML removes that bias, completely in
    balanced cases.

    The two are NOT interchangeable for testing. ML likelihoods can be
    compared across different mean structures; REML likelihoods cannot,
    because they are likelihoods of different data. ``comparable_across
    _mean_models`` records which regime the fit is in rather than
    leaving it to be remembered.

    ``method='both'`` fits both and returns the pair, which is the
    cheapest way to see how much the mean cost.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : array-like, shape (n,)
    variogram_model : {'exponential', 'spherical', 'gaussian', 'linear'}
    method : {'ml', 'reml', 'both'}
    X : array-like, optional
        Mean design matrix. A constant mean by default.

    Returns
    -------
    RichResult
        ``parameters``, ``nugget``, ``psill``, ``range``, ``sill``,
        ``beta``, ``neg2loglik``, ``converged``, and for ``'both'``
        also ``ml`` and ``reml`` sub-results.

    References
    ----------
    Schabenberger and Gotway (2005), sections 4.5.2 and 5.5.2-5.5.3,
    equations (4.35)-(4.40), pp. 166-169 and p. 263.
    Patterson and Thompson (1971). Harville (1974).
    Mardia and Marshall (1984), *Biometrika* 71:135-146.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> co = rng.uniform(0, 10, size=(50, 2))
    >>> z = rng.normal(size=50)
    >>> out = schabenberger_ml_variogram(co, z, method="reml")
    >>> bool(out["sill"] > 0)
    True
    """
    model = variogram_model
    if model not in MODELS:
        raise ValueError("model must be one of %s, got %r." % (MODELS, model))
    if method not in ("ml", "reml", "both"):
        raise ValueError(
            "method must be 'ml', 'reml' or 'both', got %r." % method
        )
    zz = np.asarray(z, dtype=float).ravel()
    v0 = float(np.var(zz))
    if v0 <= 0:
        raise ValueError("z has zero variance; no covariance to estimate.")

    def _fit(kind):
        obj_fn = gaussian_neg2loglik if kind == "ml" else reml_neg2loglik

        def obj(p):
            t = np.exp(p)
            val, _ = obj_fn(coords, zz, model, t[0], t[1], t[2], X)
            return val if np.isfinite(val) else 1e12

        start = np.log(np.array([0.1 * v0, 0.9 * v0, 1.0]))
        # a crude range sweep first: the profile in the range parameter
        # is often flat-then-steep, and a simplex started in the flat
        # part converges to the starting value and calls it a fit
        best, bestval = None, np.inf
        from ._schaben import pair_differences
        h, _ = pair_differences(coords, zz)
        for frac in (0.1, 0.25, 0.5, 1.0):
            s = start.copy()
            s[2] = np.log(max(frac * float(np.max(h)), 1e-6))
            p, val = _nelder_mead(obj, s)
            if val < bestval:
                best, bestval = p, val
        t = np.exp(best)
        val, beta = obj_fn(coords, zz, model, t[0], t[1], t[2], X)
        return {
            "nugget": float(t[0]), "psill": float(t[1]),
            "range": float(t[2]), "sill": float(t[0] + t[1]),
            "beta": np.atleast_1d(beta), "neg2loglik": float(val),
            "converged": bool(np.isfinite(val)),
        }

    if method == "both":
        ml, reml = _fit("ml"), _fit("reml")
        payload = dict(reml)
        payload.update({"ml": ml, "reml": reml, "method_used": "both"})
    else:
        payload = dict(_fit(method))
        payload["method_used"] = method

    payload.update({
        "estimate": np.array([payload["nugget"], payload["psill"],
                              payload["range"]]),
        "parameters": {"nugget": payload["nugget"], "psill": payload["psill"],
                       "range": payload["range"]},
        "model": model,
        "comparable_across_mean_models": payload["method_used"] == "ml",
        "comparison_note": (
            "ML likelihoods may be compared across different mean "
            "structures; REML ones may not, being likelihoods of different "
            "data (the error contrasts)"
        ),
        "bias_note": (
            "ML covariance estimates are biased downward because the "
            "degrees of freedom spent on the mean are unaccounted for; for "
            "independent data with unknown mean the bias is exactly -theta/n"
        ),
        "n": int(zz.size),
        "method": "%s estimation of covariance parameters"
                  % payload["method_used"].upper(),
    })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "spml: covariance parameters by ML (4.35) or REML (4.39), with the "
        "downward ML bias and the likelihood-comparison rule both stated"
    )
