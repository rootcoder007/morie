"""Nonparametric (spectral step-function) semivariogram, Shapiro and Botha."""

from ._richresult import RichResult
from ._schab_npvg import omega
from .nnlsq import nonnegative_least_squares

__all__ = ["shapiro_botha_semivariogram"]


def shapiro_botha_semivariogram(h, nodes, weights=None, d=2, gamma_hat=None, npairs=None):
    r"""Nonparametric semivariogram from a step-function spectral measure.

    Schabenberger & Gotway (2005, Sec. 4.3.4.1 and 4.6.1.1, eqs 4.16, 4.46,
    4.47): giving the isotropic spectral measure mass :math:`w_i \ge 0` at
    nodes :math:`t_i`,

    .. math::

        C(h, t, w) = \sum_{i=1}^p w_i\,\Omega_d(h t_i),\qquad
        \gamma(h, t, w) = \sum_{i=1}^p w_i\,\{1 - \Omega_d(h t_i)\},

    with :math:`\Omega_1 = \cos`, :math:`\Omega_2 = J_0`,
    :math:`\Omega_3(t) = \sin t/t`. Any non-negative weights give a valid
    model in :math:`R^d`, and it can take negative correlations.

    With ``weights`` given the model is evaluated at ``h``. With
    ``gamma_hat`` given instead, ``h`` are the lags of an empirical
    semivariogram and the weights are fitted by least squares under
    :math:`w \ge 0` (Shapiro & Botha 1991), weighting lag :math:`k` by
    ``npairs[k]`` when supplied.

    Parameters
    ----------
    h : sequence of float
        Lags.
    nodes : sequence of float
        Nodes :math:`t_i`, non-negative.
    weights : sequence of float, optional
        Masses :math:`w_i \ge 0` (evaluate mode).
    d : int
        Dimension, 1, 2 or 3.
    gamma_hat : sequence of float, optional
        Empirical semivariogram at ``h`` (fit mode).
    npairs : sequence of float, optional
        Pair counts for weighted least squares (fit mode).

    Returns
    -------
    RichResult
        ``gamma`` and ``covariance`` at ``h``, ``weights``, ``sill``
        (:math:`\sum w_i`), and in fit mode ``rss``.

    References
    ----------
    Shapiro, A. & Botha, J. D. (1991). Variogram fitting with a general
    class of conditionally nonnegative definite functions. Computational
    Statistics and Data Analysis 11, 87-96. Schabenberger, O. & Gotway,
    C. A. (2005). Statistical Methods for Spatial Data Analysis. Chapman &
    Hall/CRC, eqs (4.16), (4.46), (4.47), pp. 147, 180.
    """
    h = [float(v) for v in h]
    t = [float(v) for v in nodes]
    if any(v < 0 for v in t) or not t:
        raise ValueError("`nodes` must be non-negative and non-empty")
    basis = [omega(d, [hv * tv for tv in t]) for hv in h]
    rss = None
    if gamma_hat is not None:
        g = [float(v) for v in gamma_hat]
        if len(g) != len(h):
            raise ValueError("`gamma_hat` must match `h`")
        sw = [1.0] * len(h) if npairs is None else [float(v) ** 0.5 for v in npairs]
        A = [[sw[k] * (1.0 - basis[k][i]) for i in range(len(t))] for k in range(len(h))]
        fit = nonnegative_least_squares(A, [sw[k] * g[k] for k in range(len(h))])
        w = fit["x"]
        rss = fit["residual_norm"] ** 2
    elif weights is not None:
        w = [float(v) for v in weights]
        if len(w) != len(t) or any(v < 0 for v in w):
            raise ValueError("`weights` must be non-negative, one per node")
    else:
        raise ValueError("give `weights` to evaluate or `gamma_hat` to fit")
    cov = [sum(wi * bi for wi, bi in zip(w, row)) for row in basis]
    sill = sum(w)
    gam = [0.0 if hv == 0 else sill - c for hv, c in zip(h, cov)]
    payload = {"gamma": gam, "covariance": cov, "weights": w, "sill": sill, "nodes": t, "d": d}
    if rss is not None:
        payload["rss"] = rss
    return RichResult(
        title="Nonparametric (Shapiro-Botha) semivariogram",
        summary_lines=[("nodes", len(t)), ("sill", sill)],
        payload=payload,
    )


def cheatsheet():
    return "spnpsv: gamma(h) = sum w_i (1 - Omega_d(h t_i)), w >= 0; fit by NNLS"
