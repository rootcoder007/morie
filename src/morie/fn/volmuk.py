# morie.fn -- function file (rootcoder007/morie)
"""Subsample-averaged realised kernel."""

import numpy as np

from ._richresult import RichResult
from .volrk import vol_realised_kernel

__all__ = ["vol_multi_kernel_rk"]


def vol_multi_kernel_rk(r_intraday, n_grids=5, H=None):
    r"""Average the realised kernel over shifted subsampling grids.

    Computes :func:`morie.fn.volrk.vol_realised_kernel` on each of
    ``n_grids`` offset subgrids (every ``n_grids``-th return, offsets
    0..n_grids-1) and averages -- the same variance-reduction trick
    subsampling brings to TSRV, applied to the kernel estimator.

    Parameters
    ----------
    r_intraday : array-like, shape (m,)
        Intraday returns at the finest grid.
    n_grids : int, default 5
        Number of offset subgrids.
    H : int, optional
        Kernel bandwidth per subgrid (default sqrt of subgrid size).

    Returns
    -------
    RichResult
        keys: ``rk_avg``, ``rk_per_grid``, ``n_grids``, ``n_returns``,
        ``method``.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. & Shephard, N.
    (2008). Designing realized kernels. *Econometrica*, 76(6),
    1481-1536.

    Zhang, L., Mykland, P. A. & Ait-Sahalia, Y. (2005). A tale of two
    time scales: determining integrated volatility with noisy
    high-frequency data. *JASA*, 100(472), 1394-1411. (subsample
    averaging)
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    g = int(n_grids)
    if g < 2:
        raise ValueError(f"n_grids must be at least 2, got {g}.")
    if r.size < 5 * g:
        raise ValueError(f"need at least {5 * g} returns for {g} grids.")

    p = np.concatenate([[0.0], np.cumsum(r)])
    vals = []
    for off in range(g):
        sub = np.diff(p[off::g])
        if sub.size >= 5:
            vals.append(vol_realised_kernel(sub, H=H)["rk"])
    if not vals:
        raise ValueError("no subgrid had enough returns.")

    return RichResult(
        payload={
            "rk_avg": float(np.mean(vals)),
            "rk_per_grid": np.array(vals),
            "n_grids": g,
            "n_returns": int(r.size),
            "method": f"Subsample-averaged realised kernel ({g} grids)",
        }
    )


def cheatsheet():
    return "volmuk: mean of realised kernels over shifted subgrids"
