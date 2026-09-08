# morie.fn -- function file (rootcoder007/morie)
"""Block-diagonal generalised least squares for independent subjects."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "glsblk",
    "statistical_methods_for_spatial_data_analysis_chapter_1_equation_1",
]


def glsblk(x, y, v=None):
    r"""Generalised least squares when the data split into independent blocks.

    Schabenberger & Gotway write the longitudinal model one subject at a
    time (eq. 1.1, p. 3),

    .. math::

        \mathbf{Y}_i = \mathbf{X}_i\boldsymbol\beta + \mathbf{e}_i,
        \qquad \mathbf{e}_i \sim (\mathbf{0}, \mathbf{V}_i(\theta)),

    the subjects being independent.  Because ``Var[e]`` is then
    block-diagonal, the GLS estimator accumulates one block at a time
    (p. 4),

    .. math::

        \hat{\boldsymbol\beta} =
        \left(\sum_{i=1}^{s}\mathbf{X}_i'\mathbf{V}_i^{-1}\mathbf{X}_i
        \right)^{-1}
        \sum_{i=1}^{s}\mathbf{X}_i'\mathbf{V}_i^{-1}\mathbf{Y}_i ,

    with :math:`\mathrm{Var}[\hat{\boldsymbol\beta}] =
    (\sum_i \mathbf{X}_i'\mathbf{V}_i^{-1}\mathbf{X}_i)^{-1}`.  This is
    the computational implication the book draws from block-diagonality:
    nothing of size :math:`n \times n` is ever formed.

    Parameters
    ----------
    x : sequence of array-like
        One design matrix ``X_i`` of shape ``(n_i, p)`` per block.
    y : sequence of array-like
        One response vector ``Y_i`` of length ``n_i`` per block.
    v : sequence of array-like, optional
        One within-block covariance ``V_i`` of shape ``(n_i, n_i)`` per
        block.  Identity matrices when omitted, which makes this OLS.

    Returns
    -------
    RichResult
        ``beta``, ``vcov``, ``se``, ``residuals``, ``n``, ``n_blocks``,
        ``p``, ``block_sizes``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eq. (1.1), p. 3, and the
    stacked estimator on p. 4.
    """
    blocks_x = [np.asarray(xi, dtype=float) for xi in x]
    blocks_x = [xi if xi.ndim == 2 else xi.reshape((xi.size, 1)) for xi in blocks_x]
    blocks_y = [np.asarray(yi, dtype=float).ravel() for yi in y]
    if len(blocks_x) == 0:
        raise ValueError("need at least one block")
    if len(blocks_x) != len(blocks_y):
        raise ValueError("`x` and `y` must have the same number of blocks")
    p = int(blocks_x[0].shape[1])
    if v is None:
        blocks_v = [np.eye(int(yi.size)) for yi in blocks_y]
    else:
        blocks_v = [np.asarray(vi, dtype=float) for vi in v]
        if len(blocks_v) != len(blocks_x):
            raise ValueError("`v` must have one matrix per block")

    xtvx = np.zeros((p, p))
    xtvy = np.zeros(p)
    for xi, yi, vi in zip(blocks_x, blocks_y, blocks_v):
        ni = int(yi.size)
        if xi.shape[0] != ni:
            raise ValueError("block design and response lengths disagree")
        if int(xi.shape[1]) != p:
            raise ValueError("all blocks must have the same number of columns")
        if vi.shape != (ni, ni):
            raise ValueError(f"block covariance must be ({ni}, {ni})")
        vinv_x = np.linalg.solve(vi, xi)
        vinv_y = np.linalg.solve(vi, yi)
        xtvx = xtvx + xi.T @ vinv_x
        xtvy = xtvy + xi.T @ vinv_y

    vcov = np.linalg.inv(xtvx)
    beta = vcov @ xtvy
    resid = []
    for xi, yi in zip(blocks_x, blocks_y):
        resid.extend(list(np.asarray(yi - xi @ beta, dtype=float).ravel()))
    sizes = [int(yi.size) for yi in blocks_y]

    return RichResult(
        title="Block-diagonal GLS (Schabenberger & Gotway eq. 1.1)",
        summary_lines=[("blocks", len(sizes)), ("n", int(sum(sizes))), ("p", p)],
        payload={
            "beta": beta,
            "vcov": vcov,
            "se": np.sqrt(np.diag(vcov)),
            "residuals": np.asarray(resid, dtype=float),
            "n": int(sum(sizes)),
            "n_blocks": len(sizes),
            "p": p,
            "block_sizes": sizes,
        },
    )


statistical_methods_for_spatial_data_analysis_chapter_1_equation_1 = glsblk


def cheatsheet():
    return "glsblk: block-diagonal GLS, beta = (sum Xi'Vi^-1 Xi)^-1 sum Xi'Vi^-1 Yi."
