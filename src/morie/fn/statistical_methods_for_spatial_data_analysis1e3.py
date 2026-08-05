# morie.fn -- function file (rootcoder007/morie)
"""Stacked block-diagonal GLS (Schabenberger & Gotway eq. 1.3)."""

from . import _array_core as np

from ._richresult import RichResult
from .statistical_methods_for_spatial_data_analysis1e1 import glsblk

__all__ = [
    "glsstk",
    "statistical_methods_for_spatial_data_analysis_chapter_1_equation_3",
]


def glsstk(x, y, subject, v=None):
    r"""GLS on the stacked model with a block-diagonal error covariance.

    Stacking the per-subject models of eq. (1.1) gives (eq. 1.3, p. 4)

    .. math::

        \mathbf{Y} = \mathbf{X}_l\boldsymbol\beta + \mathbf{e},
        \qquad
        \mathrm{Var}[\mathbf{e}] = \mathbf{V}(\theta_l)
        = \mathrm{blockdiag}(\mathbf{V}_1, \dots, \mathbf{V}_s).

    The estimator is identical to the accumulated one; this entry point
    simply takes the data already stacked, splits it on ``subject`` and
    delegates to :func:`glsblk`.  Blocks are formed in order of first
    appearance of the subject label, so the caller controls the order.

    Parameters
    ----------
    x : array-like
        Stacked design matrix, shape ``(n, p)``.
    y : array-like
        Stacked response, length ``n``.
    subject : array-like
        Block label per row.
    v : sequence of array-like, optional
        One ``V_i`` per block, in order of first appearance.

    Returns
    -------
    RichResult
        Same keys as :func:`glsblk`, plus ``labels``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eq. (1.3), p. 4.
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape((X.size, 1))
    z = np.asarray(y, dtype=float).ravel()
    n = int(z.size)
    if int(X.shape[0]) != n:
        raise ValueError("`x` and `y` must have the same number of rows")
    lab = list(subject)
    if len(lab) != n:
        raise ValueError("`subject` must have one label per row")

    order = []
    for s in lab:
        if s not in order:
            order.append(s)
    idx = [[i for i in range(n) if lab[i] == s] for s in order]
    bx = [np.asarray([[float(X[i, j]) for j in range(int(X.shape[1]))] for i in ii],
                     dtype=float) for ii in idx]
    by = [np.asarray([float(z[i]) for i in ii], dtype=float) for ii in idx]
    res = glsblk(bx, by, v)
    payload = dict(res)
    payload["labels"] = order
    return RichResult(
        title="Stacked block-diagonal GLS (Schabenberger & Gotway eq. 1.3)",
        summary_lines=[("blocks", len(order)), ("n", n), ("p", payload["p"])],
        payload=payload,
    )


statistical_methods_for_spatial_data_analysis_chapter_1_equation_3 = glsstk


def cheatsheet():
    return "glsstk: stacked GLS with block-diagonal V; splits on subject and calls glsblk."
