"""Curds and whey: canonical-correlation shrinkage of multiple-response regression (ESL sec 3.7)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["esl_curds_whey"]


def _inv_sqrt(S):
    w, V = np.linalg.eigh(np.asarray(S, dtype=float))
    if float(np.min(w)) <= 0:
        raise ValueError("a covariance matrix is singular")
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def esl_curds_whey(X, Y, lambda_=0.0, newdata=None):
    r"""Breiman-Friedman curds-and-whey fit :math:`\hat Y^{c+w} = HYS^{c+w}` (ESL eqs 3.72-3.75).

    With the canonical correlations :math:`c_m` and Y-side canonical vectors
    U of the centred X and Y, :math:`S^{c+w} = U\Lambda U^{-1}` with
    :math:`\lambda_m = c_m^2/(c_m^2 + (p/N)(1-c_m^2))` (eq 3.73), so each
    canonical response direction is shrunk by how poorly it is predicted.
    ``lambda_ > 0`` gives the hybrid of eq 3.75, :math:`A_\lambda YS^{c+w}` with the
    ridge operator :math:`A_\lambda = X(X^TX+\lambda I)^{-1}X^T` on the centred X.
    Intercepts are the column means. Needs K <= p responses.

    Parameters
    ----------
    X : N x p nested sequence
    Y : N x K nested sequence
    lambda_ : float
        Ridge penalty for the hybrid (0 = least squares).
    newdata : M x p nested sequence, optional

    Returns
    -------
    RichResult
        ``coefficients`` (p x K, on the centred scale), ``intercept`` (K),
        ``fitted``, ``prediction``, ``canonical_correlations``,
        ``shrinkage`` (lambda_m), ``S`` (K x K).

    References
    ----------
    Breiman, L. & Friedman, J. (1997). Predicting multivariate responses in
    multiple linear regression. JRSS B 59, 3-54.
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    Ya = np.atleast_2d(np.asarray(Y, dtype=float))
    N, p = Xa.shape
    K = Ya.shape[1]
    if Ya.shape[0] != N or p < K or p >= N:
        raise ValueError("need N > p rows in X and Y and K <= p responses")
    xm, ym = Xa.mean(axis=0), Ya.mean(axis=0)
    Xc, Yc = Xa - xm, Ya - ym
    Sxx, Syy, Sxy = Xc.T @ Xc / N, Yc.T @ Yc / N, Xc.T @ Yc / N
    Rx, Ry = _inv_sqrt(Sxx), _inv_sqrt(Syy)
    Uw, c, Vt = np.linalg.svd(Rx @ Sxy @ Ry, full_matrices=False)
    U = Ry @ Vt.T  # Y-side canonical vectors (K x K)
    cs = [float(v) for v in c]
    lam = [v * v / (v * v + (p / N) * (1 - v * v)) for v in cs]
    S = U @ np.diag(np.asarray(lam)) @ np.linalg.inv(U)
    G = Xc.T @ Xc + float(lambda_) * np.eye(p)
    Bhat = np.linalg.solve(G, Xc.T @ Yc)
    B = Bhat @ S
    fitted = Xc @ B + ym
    pred = None
    if newdata is not None:
        pred = ((np.atleast_2d(np.asarray(newdata, dtype=float)) - xm) @ B + ym).tolist()
    return RichResult(
        title="Curds and whey",
        summary_lines=[("canonical_correlations", cs)],
        payload={
            "coefficients": B.tolist(),
            "intercept": [float(v) for v in ym - xm @ B],
            "fitted": fitted.tolist(),
            "prediction": pred,
            "canonical_correlations": cs,
            "shrinkage": lam,
            "S": S.tolist(),
        },
    )


def cheatsheet():
    return "eslcaw: S = U diag(c^2 / (c^2 + (p/N)(1 - c^2))) U^-1 from the CCA; Y_hat = H Y S (ridge: A_lambda Y S)"
