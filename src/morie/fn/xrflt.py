"""Eigenvector spatial filtering"""

from .sfilter import eigenvector_filtering


def spatial_filter(data, W, X=None, *, method="moran", tol=0.1):
    r"""Eigenvector spatial filtering by forward selection of Moran eigenvectors.

    Thin front-end to :func:`morie.fn.sfilter.eigenvector_filtering`:
    data is the response, W the spatial weights, method the
    selection criterion (moran: stop when the residual \|Moran's I\| is
    below tol, Tiefelsdorf and Griffith 2007; aic, press or
    r2, Griffith 2003).

    References
    ----------
    Tiefelsdorf, M. and Griffith, D. A. (2007). Semiparametric filtering of
    spatial autocorrelation: the eigenvector approach. *Environment and
    Planning A* 39, 1193-1221.
    Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*.
    Springer.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(8)] for i in range(8)]
    >>> r = spatial_filter([1.0, 2.0, 2.5, 4.0, 3.5, 3.0, 1.5, 1.0], W, method="aic")
    >>> r["selected"]
    [1]
    """
    return eigenvector_filtering(data, W, X, criterion=method, tol=tol)


spat = spatial_filter


def cheatsheet() -> str:
    return "spatial_filter(y, W, method='moran') -> Moran eigenvector spatial filtering (forward selection)."
