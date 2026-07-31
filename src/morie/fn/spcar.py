"""Conditional autoregressive (CAR) model (delegates to sgcar)."""

from .sgcar import conditional_autoregressive

__all__ = ["schabenberger_car_model"]


def schabenberger_car_model(z, w, covariates=None):
    r"""
    Conditional autoregressive model: the conditional specification.

    Rather than specifying one multivariate model, the CAR approach
    models each conditional distribution
    :math:`f(Z(s_i) \mid Z(s_j), s_j \in N_i)`:

    .. math::

        E[Z(s_i) \mid Z(s)_{-i}] = x(s_i)'\beta
            + \sum_j c_{ij}(Z(s_j) - x(s_i)'\beta),
        \qquad \mathrm{Var}[Z(s_i) \mid Z(s)_{-i}] = \sigma_i^2

    The Hammersley-Clifford theorem gives the conditions under which
    those conditionals define a valid joint distribution; in the Gaussian
    case they do, with :math:`\Sigma_{CAR} = (I - C)^{-1}\Sigma_c`.

    Same estimator as
    :func:`morie.fn.sgcar.conditional_autoregressive`; this delegates
    rather than carrying a second implementation.

    Parameters
    ----------
    z : array-like
        Response, shape (n,).
    w : array-like
        Adjacency weights, shape (n, n).
    covariates : array-like, optional
        Covariates; an intercept when omitted.

    Returns
    -------
    The result of ``conditional_autoregressive``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.2.2.2, eqs.
    (6.43)-(6.45), pp. 338-339.
    """
    return conditional_autoregressive(z, w, covariates)


def cheatsheet():
    return "spcar: CAR model; delegates to conditional_autoregressive (sgcar)."
