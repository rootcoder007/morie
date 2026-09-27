"""Supervised principal components (ESL Alg. 18.1)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["esl_supervised_pc"]


def esl_supervised_pc(X, y, threshold, m=1, newdata=None):
    r"""Regress y on the leading principal components of the features most associated with it.

    ESL Alg. 18.1 (Bair, Hastie, Paul & Tibshirani 2006): with centred data,
    the standardized univariate coefficient of feature j is
    :math:`s_j = x_j^Ty/\|x_j\|`; keep the features with :math:`|s_j| >
    \theta`, take the first m principal components of that reduced matrix
    (SVD of the centred columns), and regress y on them by least squares.
    Under the latent-variable model of eqs 18.32-18.33 the leading component
    estimates the latent U. New data are projected with the same loadings.
    The threshold is chosen by cross-validation in practice.

    Parameters
    ----------
    X : N x p nested sequence
    y : sequence of N floats
    threshold : float
        theta >= 0.
    m : int
        Number of components.
    newdata : M x p nested sequence, optional

    Returns
    -------
    RichResult
        ``scores_univariate`` (s_j), ``selected`` (0-based), ``loadings``
        (p_selected x m), ``coefficients`` (intercept then one per
        component), ``fitted``, ``prediction``.

    References
    ----------
    Bair, E., Hastie, T., Paul, D. & Tibshirani, R. (2006). Prediction by
    supervised principal components. JASA 101, 119-137.
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    ya = np.asarray(y, dtype=float).ravel()
    N, p = Xa.shape
    if ya.size != N or threshold < 0 or m < 1:
        raise ValueError("need matching X and y, threshold >= 0 and m >= 1")
    xm, ym = Xa.mean(axis=0), float(ya.mean())
    Xc, yc = Xa - xm, ya - ym
    s = [float(Xc[:, j] @ yc) / math.sqrt(float(Xc[:, j] @ Xc[:, j])) for j in range(p)]
    sel = [j for j in range(p) if abs(s[j]) > threshold]
    if len(sel) < m:
        raise ValueError(f"only {len(sel)} features pass the threshold; need at least m = {m}")
    Xs = Xc[:, sel]
    U, D, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt.T[:, :m]
    for k in range(m):
        i = int(np.argmax(np.abs(V[:, k])))
        if V[i, k] < 0:
            V[:, k] = -V[:, k]
    Z = Xs @ V
    beta = np.linalg.solve(Z.T @ Z, Z.T @ yc)
    fitted = (Z @ beta + ym).tolist()
    pred = None
    if newdata is not None:
        Nd = np.atleast_2d(np.asarray(newdata, dtype=float)) - xm
        pred = (Nd[:, sel] @ V @ beta + ym).tolist()
    return RichResult(
        title="Supervised principal components",
        summary_lines=[("selected", len(sel))],
        payload={
            "scores_univariate": s,
            "selected": sel,
            "loadings": V.tolist(),
            "coefficients": [ym] + [float(v) for v in beta],
            "fitted": fitted,
            "prediction": pred,
        },
    )


def cheatsheet():
    return "eslsup: s_j = x_j'y / |x_j|; PCA of features with |s_j| > theta; regress y on the first m components"
