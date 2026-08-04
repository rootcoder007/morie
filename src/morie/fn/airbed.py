# morie.fn -- function file (rootcoder007/morie)
"""Activity-data times emission-factor inventory."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["emisinv", "emissions_inventory"]


def emisinv(activity, factor, gwp=None):
    """Greenhouse-gas inventory from activity data and emission factors.

    The IPCC inventory equation combines a measure of how much of a human
    activity takes place with a coefficient giving emissions per unit of
    that activity:

        Emissions = AD * EF

    summed over the cells of a sector-by-fuel (or sector-by-gas) table.
    Where the cells are different gases, each is put on a common footing
    by its global warming potential before summation.

    Parameters
    ----------
    activity : array-like, shape (s, f) or (f,)
        Activity data, one entry per sector-fuel cell.
    factor : array-like, same shape as activity
        Emission factors in mass per unit of activity.
    gwp : array-like or None
        Per-column global warming potentials.  ``None`` leaves the cell
        emissions in their own mass units.

    Returns
    -------
    RichResult
        ``total``, ``cell``, ``bysector``, ``byfuel``, ``s``, ``f``.

    References
    ----------
    IPCC (2006), 2006 IPCC Guidelines for National Greenhouse Gas
    Inventories, Volume 1, Chapter 1 (Introduction), Sect. 1.2: "The
    basic equation is therefore: Emissions = AD * EF", where AD is
    activity data and EF an emission factor.  Read from the official PDF
    at www.ipcc-nggip.iges.or.jp.
    """
    A = C.mat(activity)
    E = C.mat(factor)
    if len(A) != len(E) or len(A[0]) != len(E[0]):
        raise ValueError("activity and factor must have the same shape")
    s, f = len(A), len(A[0])
    if gwp is None:
        g = [1.0] * f
    else:
        g = C.vec(gwp)
        if len(g) != f:
            raise ValueError("gwp must have one entry per column")
    cell = [[A[i][j] * E[i][j] * g[j] for j in range(f)] for i in range(s)]
    return RichResult(payload={
        "total": sum(sum(r) for r in cell), "cell": cell,
        "bysector": [sum(r) for r in cell],
        "byfuel": [sum(cell[i][j] for i in range(s)) for j in range(f)],
        "s": s, "f": f,
        "method": "IPCC inventory equation Emissions = AD * EF"})


emissions_inventory = emisinv


def cheatsheet():
    return "airbed: Activity-data times emission-factor inventory."
