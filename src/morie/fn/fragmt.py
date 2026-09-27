"""Landscape fragmentation: effective mesh size, splitting index and landscape division (Jaeger 2000).

Jaeger, J. A. G. (2000). Landscape division, splitting index, and effective mesh size: new
measures of landscape fragmentation. Landscape Ecology 15, 115-130.
"""

from ._richresult import RichResult

__all__ = ["landscape_fragmentation"]


def _patches(grid, cls, eight):
    h, w = len(grid), len(grid[0])
    lab = [[-1] * w for _ in range(h)]
    sizes = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if eight else [])
    for y in range(h):
        for x in range(w):
            if grid[y][x] == cls and lab[y][x] < 0:
                lab[y][x] = len(sizes)
                stack, size = [(y, x)], 0
                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    for dy, dx in steps:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == cls and lab[ny][nx] < 0:
                            lab[ny][nx] = lab[y][x]
                            stack.append((ny, nx))
                sizes.append(size)
    return sizes, lab


def landscape_fragmentation(grid, cls=1, cell_area=1.0, eight=True):
    r"""Fragmentation of the class ``cls`` in a categorical raster.

    Patches are connected components of class cells (8-neighbour by default). With patch
    areas a_i and total landscape area A: effective mesh size m_eff = sum a_i^2 / A, splitting
    index S = A^2 / sum a_i^2 and landscape division D = 1 - sum (a_i / A)^2 (Jaeger 2000);
    these equal landscapemetrics' lsm_c_mesh (in the units of ``cell_area``), lsm_c_split and
    lsm_c_division.

    Parameters
    ----------
    grid : 2-D list of class codes
    cls : class code to measure
    cell_area : float
    eight : bool
        8-neighbour (True) or 4-neighbour connectivity.

    Returns
    -------
    RichResult
        Keys: mesh, splitting, division, n_patches, patch_areas, labels.

    References
    ----------
    Jaeger, J. A. G. (2000). Landscape Ecology 15, 115-130.

    Examples
    --------
    >>> landscape_fragmentation([[1, 1, 0, 1], [1, 1, 0, 1]])["n_patches"]
    2
    """
    G = [list(r) for r in grid]
    sizes, lab = _patches(G, cls, eight)
    A = len(G) * len(G[0]) * float(cell_area)
    sq = 0.0
    for s in sizes:
        sq += (s * cell_area) ** 2
    return RichResult(
        title="Landscape fragmentation",
        summary_lines=[("effective mesh", sq / A)],
        payload={
            "mesh": sq / A,
            "splitting": A * A / sq if sq > 0 else float("inf"),
            "division": 1 - sq / (A * A),
            "n_patches": len(sizes),
            "patch_areas": [s * cell_area for s in sizes],
            "labels": lab,
        },
    )


def cheatsheet():
    return "fragmt: effective mesh size, splitting index and landscape division (Jaeger 2000)"
