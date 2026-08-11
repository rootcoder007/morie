"""Hexagonal grid binning (Carr et al.)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hexgrd", "hexagonal_grid"]


def hexgrd(coords, values=None, cell_size=1.0):
    """
    Bin 2-d points on a hexagon lattice, Dan Carr's hbin algorithm.

    Two interleaved rectangular lattices of hexagon centers are used:
    lattice A at (j * w, i * w * sqrt(3)) and lattice B at
    ((j + 1/2) w, (i + 1/2) w sqrt(3)), for integer i, j and hexagon
    x-spacing (width) w = `cell_size`. In the scaled coordinates
    sx = (x - xmin)/w, sy = (y - ymin)/(w sqrt(3)), a point belongs to
    the nearer of its two candidate centers under the anisotropic
    distance d^2 = (sx - j)^2 + 3 (sy - i)^2, which is the Euclidean
    distance in data units (up to the common factor w^2). The fast-path
    thresholds d^2 < 1/4 (certainly lattice A) and d^2 > 1/3 (certainly
    lattice B) are Carr's `con1`/`con2` constants.

    Sources
    -------
    Carr, D. B., Littlefield, R. J., Nicholson, W. L. & Littlefield,
    J. S. (1987). Scatterplot matrix techniques for large N. *JASA*,
    82(398), 424-436 (hexagon binning).
    Carr, D. B., Olsen, A. R. & White, D. (1992). Hexagon mosaic maps
    for display of univariate and bivariate geographical data.
    *Cartography and Geographic Information Systems*, 19(4), 228-236.
    Reference implementation: Dan Carr's Fortran subroutine `hbin`
    (1991, version 1994-09-16), src/hbin.f in CRAN hexbin 1.28.6
    (fetched-wave3/carr-hexbin_1.28.6-cran-source.tar.gz); the scaled
    nearest-center arithmetic above is transcribed from it.

    Parameters
    ----------
    coords : array-like, (n, 2)
        Point locations.
    values : array-like, (n,), optional
        Values to aggregate per cell (mean); None bins counts only.
    cell_size : float
        Hexagon width w (center spacing along x), in data units.

    Returns
    -------
    RichResult
        Keys: cell_id (per point), centers (unique cell centers, data
        units), counts, xcm/ycm (within-cell means of the coordinates),
        value_mean (per cell, if `values` given).
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.shape[1] != 2:
        raise ValueError("`coords` must be (n, 2)")
    n = coords.shape[0]
    w = float(cell_size)
    if w <= 0:
        raise ValueError("`cell_size` must be positive")
    vals = None
    if values is not None:
        vals = np.asarray(values, dtype=float).ravel()
        if vals.size != n:
            raise ValueError("`values` must match `coords` length")
    xmin, ymin = float(coords[:, 0].min()), float(coords[:, 1].min())
    sx = (coords[:, 0] - xmin) / w
    sy = (coords[:, 1] - ymin) / (w * np.sqrt(3.0))
    con1, con2 = 0.25, 1.0 / 3.0
    keys = []
    for i in range(n):
        sxi, syi = float(sx[i]), float(sy[i])
        j1, i1 = int(np.floor(sxi + 0.5)), int(np.floor(syi + 0.5))
        d1 = (sxi - j1) ** 2 + 3.0 * (syi - i1) ** 2
        if d1 < con1:
            key = (j1, i1, 0)
        else:
            j2, i2 = int(np.floor(sxi)), int(np.floor(syi))
            d2 = (sxi - j2 - 0.5) ** 2 + 3.0 * (syi - i2 - 0.5) ** 2
            if d1 > con2 or d1 > d2:
                key = (j2, i2, 1)
            else:
                key = (j1, i1, 0)
        keys.append(key)
    uniq = sorted(set(keys))
    index = {k: c for c, k in enumerate(uniq)}
    cell_id = np.asarray([index[k] for k in keys], dtype=int)
    m = len(uniq)
    counts = np.zeros(m, dtype=int)
    xcm = np.zeros(m)
    ycm = np.zeros(m)
    vsum = np.zeros(m)
    for i, c in enumerate(cell_id):
        c = int(c)
        counts[c] += 1
        xcm[c] += float(coords[i, 0])
        ycm[c] += float(coords[i, 1])
        if vals is not None:
            vsum[c] += float(vals[i])
    xcm = xcm / counts
    ycm = ycm / counts
    centers = np.zeros((m, 2))
    for c, (j, i, off) in enumerate(uniq):
        centers[c, 0] = xmin + (j + 0.5 * off) * w
        centers[c, 1] = ymin + (i + 0.5 * off) * w * np.sqrt(3.0)
    payload = {
        "cell_id": cell_id, "centers": centers, "counts": counts,
        "xcm": xcm, "ycm": ycm, "cell_size": w, "n": int(n),
        "method": "Carr hexagon binning (hbin transcription)",
    }
    if vals is not None:
        payload["value_mean"] = vsum / counts
    return RichResult(payload=payload)


# long descriptive alias (stub-era name)
hexagonal_grid = hexgrd


def cheatsheet():
    return "hexgrd: Carr hbin hexagon binning; nearest of two offset lattices"
