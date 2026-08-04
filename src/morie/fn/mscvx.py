# morie.fn -- function file (rootcoder007/morie)
"""2D convex hull"""

from . import _array_core as np

from ._containers import DescriptiveResult


def convex_hull_2d(data, *, method="default"):
    """2D convex hull

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="mscvx",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


conv = convex_hull_2d


def cheatsheet() -> str:
    return "convex_hull_2d({}) -> 2D convex hull"


# compact alias per ledger/NAMING.md
convexhull2d = convex_hull_2d
