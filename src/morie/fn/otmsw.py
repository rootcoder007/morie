# morie.fn -- function file (rootcoder007/morie)
"""Max-sliced Wasserstein distance."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_max_sliced_w"]


def ot_max_sliced_w(X, Y, p=2, n_proj=32):
    """Take the worst direction rather than the average one.

    Averaging over directions wastes most of the budget on slices where
    the two clouds already agree; in high dimension almost every random
    direction is such a slice, and the sliced distance decays.  Keeping
    only the maximising direction gives a sharper discrepancy and, as a
    by-product, the direction itself -- an interpretable statement about
    where the two clouds differ.

    Formula: ``max_theta W_p(P_theta mu, P_theta nu)`` over unit
    directions -- Deshpande et al. (2019) eq. (3).

    Parameters
    ----------
    X, Y : array-like, shape (n, d)
        Two point clouds with the same number of points.
    p : float, default 2
        Exponent, positive.
    n_proj : int, default 32
        Number of candidate directions searched.

    Returns
    -------
    RichResult
        ``MSW``, ``theta_star``, ``idx_star``, ``per_proj``, ``n``,
        ``d``, ``n_proj``.

    References
    ----------
    Deshpande, I., Hu, Y.-T., Sun, R., Pyrros, A., Siddiqui, N., Koyejo,
    S., Zhao, Z., Forsyth, D. and Schwing, A. (2019).  Max-sliced
    Wasserstein distance and its use for GANs.  Proceedings of the IEEE
    Conference on Computer Vision and Pattern Recognition, 10640-10648.
    doi:10.1109/CVPR.2019.01090.
    """
    A = core.mat(X)
    B = core.mat(Y)
    if len(A) != len(B):
        raise ValueError("max-sliced W_p needs clouds with equal point counts")
    d = len(A[0])
    if len(B[0]) != d:
        raise ValueError("point clouds must share a dimension")
    pp = float(p)
    L = int(n_proj)
    TH = ot.directions(d, L)
    per = [ot.wp1d(ot.project(A, th), ot.project(B, th), pp) for th in TH]
    best = 0
    for k in range(1, L):
        if per[k] > per[best]:
            best = k
    return RichResult(payload={
        "MSW": per[best], "theta_star": TH[best], "idx_star": best,
        "per_proj": per, "n": len(A), "d": d, "n_proj": L,
        "method": "Max-sliced Wasserstein distance"})


def cheatsheet():
    return "otmsw: max-sliced Wasserstein distance and its worst direction"


# compact alias per ledger/NAMING.md
otmaxslicedw = ot_max_sliced_w
