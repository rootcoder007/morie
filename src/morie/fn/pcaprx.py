# morie.fn -- function file (rootcoder007/morie)
"""Principal Component Analysis with R-style verbose result."""

from collections.abc import Sequence
from typing import Union

from . import _array_core as np
from ._ml_core import PCA


def pcaprx(X: Union[Sequence, np.ndarray], n_components: int | None = None, standardize: bool = True):
    r"""Principal component analysis by eigendecomposition of the covariance.

    Columns are optionally standardised (sample standard deviation;
    constant columns left unscaled), the sample covariance ``S = X_c'X_c /
    (n - 1)`` is decomposed, and the components are its eigenvectors in
    decreasing eigenvalue order, each signed so that its largest-magnitude
    loading is positive; scores are ``X_c V`` (Jolliffe 2002, ch. 1-3).
    ``n_for_80`` and ``n_for_95`` are the smallest numbers of components
    whose cumulative variance share reaches 80% and 95%.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Data.
    n_components : int, optional
        Components retained (all by default).
    standardize : bool
        Correlation-matrix PCA (True) or covariance PCA.

    Returns
    -------
    RichResult
        ``components`` (rows), ``scores``, ``explained_variance``,
        ``explained_variance_ratio``, ``n_for_80``, ``n_for_95``.

    References
    ----------
    Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. Springer.

    Examples
    --------
    >>> X = [[2.5, 2.4, 1.0], [0.5, 0.7, 2.1], [2.2, 2.9, 0.4], [1.9, 2.2, 1.8], [3.1, 3.0, 0.9]]
    >>> r = pcaprx(X)
    >>> [round(v, 8) for v in r["explained_variance"]], r["n_for_80"]
    ([2.6968172, 0.26409782, 0.03908498], 1)
    """
    from ._richresult import RichResult

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
    else:
        Xs = X
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_for_80 = int(np.argmax(cum_var >= 0.80) + 1) if (cum_var >= 0.80).any() else len(cum_var)
    n_for_95 = int(np.argmax(cum_var >= 0.95) + 1) if (cum_var >= 0.95).any() else len(cum_var)
    rows = [
        [f"PC{i + 1}", f"{ev:.4g}", f"{evr * 100:.2f}%", f"{cum * 100:.2f}%"]
        for i, (ev, evr, cum) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_, cum_var))
    ]
    return RichResult(
        title="Principal Component Analysis",
        summary_lines=[
            ("n observations", n),
            ("p variables", p),
            ("Components retained", len(pca.explained_variance_)),
            ("Standardized", standardize),
            ("Components for >=80% variance", n_for_80),
            ("Components for >=95% variance", n_for_95),
        ],
        tables=[
            {
                "title": "Variance explained per component:",
                "headers": ["PC", "Eigenvalue", "Var ratio", "Cumulative"],
                "rows": rows,
            }
        ],
        warnings=[]
        if standardize or all(X.var(axis=0, ddof=1) > 1e-10)
        else [
            "raw scaling: variables on different scales dominate PCA. Pass standardize=True unless you have a reason."
        ],
        payload={
            "components": pca.components_.tolist(),
            "scores": scores.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_for_80": n_for_80,
            "n_for_95": n_for_95,
        },
    )


def cheatsheet() -> str:
    return "pcaprx: pcaprx(X, n_components, standardize) -> PCA: orthogonal components ranked by explained variance."
