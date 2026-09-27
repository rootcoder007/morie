"""Varying-coefficient model by locally weighted least squares (ESL sec 6.4.2)."""

from . import _array_core as np
from ._richresult import RichResult
from .eslnnk import _KERNELS, _kernel_weights
from .linsys import _householder_ls

__all__ = ["esl_varying_coef"]


def esl_varying_coef(X, z, y, z0, lambda_, kernel="epanechnikov"):
    r"""Fit :math:`f(X) = \alpha(Z) + \beta_1(Z)X_1 + \dots + \beta_q(Z)X_q` at each :math:`z_0`.

    ESL eqs 6.16-6.17: for each target :math:`z_0` minimise
    :math:`\sum_i K_\lambda(z_0, z_i)(y_i - \alpha(z_0) - x_i^T\beta(z_0))^2`, a
    weighted least-squares fit with kernel weights on :math:`|z_i - z_0|/\lambda`.

    Parameters
    ----------
    X : n x q nested sequence
    z : sequence of n floats
        The variable the coefficients vary with.
    y : sequence of n floats
    z0 : sequence of floats
        Target points.
    lambda_ : float
        Bandwidth, > 0.
    kernel : {"epanechnikov", "tri-cube", "gaussian"}

    Returns
    -------
    RichResult
        ``coefficients`` (one row per z0: alpha then beta), ``n_in_window``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 6.4.2.
    """
    rows = [[float(v) for v in r] for r in X]
    zz = [float(v) for v in z]
    yy = [float(v) for v in y]
    n, q = len(rows), len(rows[0])
    if len(zz) != n or len(yy) != n or lambda_ <= 0 or kernel not in _KERNELS:
        raise ValueError("need X, z and y of equal length, lambda > 0 and a known kernel")
    coefs, cnt = [], []
    for t in z0:
        w = [float(v) for v in _kernel_weights(np.asarray([abs(float(t) - v) / lambda_ for v in zz]), kernel)]
        keep = [i for i in range(n) if w[i] > 0]
        cnt.append(len(keep))
        if len(keep) <= q:
            coefs.append([float("nan")] * (q + 1))
            continue
        sw = [w[i] ** 0.5 for i in keep]
        A = [[s] + [s * v for v in rows[i]] for s, i in zip(sw, keep)]
        coefs.append(_householder_ls(A, [s * yy[i] for s, i in zip(sw, keep)])[0])
    return RichResult(
        title="Varying-coefficient model",
        summary_lines=[("targets", len(coefs))],
        payload={"coefficients": coefs, "n_in_window": cnt},
    )


def cheatsheet():
    return "eslvcm: at each z0, WLS of y on (1, X) with weights K(|z - z0| / lambda), ESL 6.17"
