"""Population density surface by quartic-kernel smoothing of point counts.

Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis. Chapman and Hall,
Sec. 4.2 (the biweight kernel). Levine, N. (2004). CrimeStat III, ch. 8 (quartic kernel
density of counts per unit area).
"""

from ._richresult import RichResult

__all__ = ["population_density_surface"]


def population_density_surface(points, counts, grid, bandwidth):
    r"""Density of people per unit area at ``grid`` locations:
    lambda(s) = sum_i c_i 3 / (pi h^2) (1 - d_is^2 / h^2)^2 over points with d_is < h.

    The quartic (biweight) kernel integrates to one over the plane, so the surface
    integrates to the total count (away from edges).

    Parameters
    ----------
    points : list of (x, y)
    counts : sequence
        Population at each point.
    grid : list of (x, y)
        Where to evaluate.
    bandwidth : float

    Returns
    -------
    RichResult
        Keys: density (per grid location), total.

    References
    ----------
    Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis, Sec. 4.2.

    Examples
    --------
    >>> import math
    >>> round(population_density_surface([(0, 0)], [10], [(0, 0)], 1.0)["density"][0] * math.pi, 12)
    30.0
    """
    import math

    h = float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")
    P = [(float(x), float(y)) for x, y in points]
    c = [float(v) for v in counts]
    out = []
    for gx, gy in grid:
        s = 0.0
        for (x, y), ci in zip(P, c):
            d2 = (gx - x) ** 2 + (gy - y) ** 2
            if d2 < h * h:
                s += ci * 3 / (math.pi * h * h) * (1 - d2 / (h * h)) ** 2
        out.append(s)
    tot = 0.0
    for v in c:
        tot += v
    return RichResult(
        title="Population density surface", summary_lines=[("total", tot)], payload={"density": out, "total": tot}
    )


def cheatsheet():
    return "popden: quartic-kernel population density surface"
