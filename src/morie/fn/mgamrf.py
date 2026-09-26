"""Multivariate gamma random field built from a shared component."""

import math

from ._richresult import RichResult

__all__ = ["multivariate_gamma_field"]


def multivariate_gamma_field(alpha, beta):
    r"""Moments of the shared-component multivariate gamma random field.

    Schabenberger & Gotway (2005, Problem 2.3, p. 79): with independent
    :math:`X_i \sim \mathrm{Gamma}(\alpha_i, \beta)`,
    :math:`E[X_i] = \alpha_i\beta`, :math:`\mathrm{Var}[X_i] = \alpha_i\beta^2`,
    let :math:`Z(s_i) = X_0 + X_i`, :math:`i = 1, \dots, n`. Then

    .. math::

        \mathrm{Cov}[Z(s_i), Z(s_j)] = \alpha_0\beta^2\ (i \ne j),\qquad
        \mathrm{Var}[Z(s_i)] = (\alpha_0 + \alpha_i)\beta^2,

        \mathrm{Corr}[Z(s_i), Z(s_j)] =
        \frac{\alpha_0}{\sqrt{(\alpha_0+\alpha_i)(\alpha_0+\alpha_j)}},

    and :math:`Z(s_i) \sim \mathrm{Gamma}(\alpha_0 + \alpha_i, \beta)`
    (sum of independent gammas with a common scale). The field is
    second-order stationary exactly when :math:`\alpha_1 = \dots =
    \alpha_n`; it is then equicorrelated, with a correlation that does
    not decay with distance.

    Parameters
    ----------
    alpha : sequence of float
        Shapes :math:`(\alpha_0, \alpha_1, \dots, \alpha_n)`, positive.
    beta : float
        Common scale, positive.

    Returns
    -------
    RichResult
        ``mean``, ``cov`` and ``corr`` (lists), ``shape`` (the marginal
        gamma shapes), ``scale``, ``stationary``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, Problem 2.3, p. 79.
    """
    a = [float(v) for v in alpha]
    beta = float(beta)
    if len(a) < 2 or any(not v > 0 for v in a) or not beta > 0:
        raise ValueError("`alpha` needs alpha_0 and at least one alpha_i, all positive; `beta` positive")
    a0, ai = a[0], a[1:]
    n = len(ai)
    mean = [(a0 + v) * beta for v in ai]
    cov = [[(a0 + ai[i]) * beta**2 if i == j else a0 * beta**2 for j in range(n)] for i in range(n)]
    corr = [[1.0 if i == j else a0 / math.sqrt((a0 + ai[i]) * (a0 + ai[j])) for j in range(n)] for i in range(n)]
    return RichResult(
        title="Multivariate gamma random field (Problem 2.3)",
        summary_lines=[("alpha_0", a0), ("sites", n), ("stationary", len(set(ai)) == 1)],
        payload={
            "mean": mean,
            "cov": cov,
            "corr": corr,
            "shape": [a0 + v for v in ai],
            "scale": beta,
            "stationary": len(set(ai)) == 1,
        },
    )


def cheatsheet():
    return "mgamrf: Z(s_i) = X_0 + X_i with gamma components; Cov = alpha_0 beta^2"
