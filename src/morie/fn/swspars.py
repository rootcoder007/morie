# morie.fn -- function file (rootcoder007/morie)
"""Sparsify weights by threshold."""

from ._qpcore import ssum


def swspars(W, thr=0.1, row_standardize=False):
    r"""Drop weak links: ``w_ij`` is set to zero when ``|w_ij| < thr``.

    Thresholding dense (e.g. inverse-distance or kernel) weights keeps the
    matrix sparse, as in the distance-decay truncation of Getis and Aldstadt
    (2004). With ``row_standardize`` the surviving weights of each row are
    rescaled to sum to one (empty rows stay zero). Returns the matrix as lists.

    References
    ----------
    Getis, A. and Aldstadt, J. (2004). Constructing the spatial weights
    matrix using a local statistic. *Geographical Analysis* 36, 90-104.

    Examples
    --------
    >>> S = swspars([[0, 0.6, 0.05], [0.6, 0, 0.3], [0.05, 0.3, 0]], thr=0.1, row_standardize=True)
    >>> [[round(v, 12) for v in r] for r in S]
    [[0.0, 1.0, 0.0], [0.666666666667, 0.0, 0.333333333333], [0.0, 1.0, 0.0]]
    """
    A = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    S = [[v if abs(v) >= thr else 0.0 for v in r] for r in A]
    if row_standardize:
        S = [[v / ssum(r) if ssum(r) != 0 else 0.0 for v in r] for r in S]
    return S


swspars_fn = swspars


def cheatsheet() -> str:
    return "swspars(W, thr=0.1, row_standardize=False) -> W with |w_ij| < thr set to zero."
