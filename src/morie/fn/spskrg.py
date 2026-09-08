"""Simple kriging: known mean mu, known C(h)."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_krig import simple_kriging

__all__ = ["schabenberger_simple_kriging"]


def schabenberger_simple_kriging(coords, z, target, cov_model=None, mu=None):
    r"""
    Simple kriging: the mean is known.

    With :math:`Z(s) = \mu(s) + e(s)`, :math:`e \sim (0, \Sigma)`, the
    predictor minimising :math:`E[(p - Z(s_0))^2]` over linear predictors
    is

    .. math::

        p_{sk}(Z; s_0) = \mu(s_0) + \sigma' \Sigma^{-1}(Z(s) - \mu(s))

    with kriging variance

    .. math::

        \sigma^2_{sk}(s_0) = \sigma^2 - \sigma' \Sigma^{-1}\sigma

    Simple kriging is an EXACT interpolator: predicting at an observed
    location returns that observation and a variance of zero, because
    :math:`\sigma` becomes a column of :math:`\Sigma`. The book calls
    this "honouring the data".

    Because no unbiasedness constraint was imposed, the predictor is best
    among ALL linear predictors -- and, if the field is Gaussian, best
    among all predictors, linear or not.

    Parameters
    ----------
    coords : array-like
        Observation coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    target : array-like
        Prediction location(s), shape ``(m, d)``.
    cov_model : mapping, optional
        ``{'model', 'nugget', 'sill', 'range'}``. Defaults to a unit-sill
        exponential with range 1.
    mu : float, optional
        The known mean. Defaults to the sample mean of ``z``, which makes
        this simple kriging with an estimated rather than known mean --
        state a value when the mean really is known.

    Returns
    -------
    RichResult
        ``prediction``, ``variance``, ``weights``, ``mu``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 5.2.1, eqs.
    (5.10)-(5.11), pp. 223-224.
    """
    pred, var, lam = simple_kriging(coords, z, target, cov_model, mu)
    mu_used = float(np.mean(np.asarray(z, dtype=float))) if mu is None else float(mu)
    return RichResult(
        title="Simple kriging",
        summary_lines=[("mu", mu_used), ("n targets", int(pred.size))],
        payload={"prediction": pred, "variance": var, "weights": lam,
                 "mu": mu_used},
    )


def cheatsheet():
    return "spskrg: simple kriging (5.10)-(5.11); an exact interpolator."
