"""Convert a soft transport plan to a barycentric map."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_plan_to_map"]


def ot_plan_to_map(T, Y):
    """
    Barycentric projection of a transport plan.

    Formula: Tbar(x_i) = sum_j T_ij y_j / sum_j T_ij

    Verified against Peyre & Cuturi (2019), Remark 4.11, eq. (4.19) --
    source consulted, which writes ``T : x_i -> (1/a_i) sum_j P_ij y_j``
    with ``a_i = sum_j P_ij``. As eps goes to zero this converges to the
    Monge map when that map is unique.

    Parameters
    ----------
    T : nested sequence
        Coupling matrix, ``n x m``.
    Y : nested sequence
        Destination points, ``m x d``.

    Returns
    -------
    RichResult
        Keys: estimate (the ``n x d`` image points), mass, displacement,
        method. ``displacement`` is the mean row mass-weighted spread.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    Remark 4.11, eq. (4.19).
    """
    Tm = _big2.mat(T)
    B = _big2.mat(Y)
    nr, nc = len(Tm), len(Tm[0])
    if len(B) != nc:
        raise ValueError("Y must have one row per column of T")
    d = len(B[0])
    out = []
    mass = []
    spread = 0.0
    for i in range(nr):
        s = sum(Tm[i])
        mass.append(s)
        if s <= 0.0:
            out.append([float("nan")] * d)
            continue
        pt = [sum(Tm[i][j] * B[j][k] for j in range(nc)) / s for k in range(d)]
        out.append(pt)
        v = 0.0
        for j in range(nc):
            w = Tm[i][j] / s
            for k in range(d):
                t = B[j][k] - pt[k]
                v += w * t * t
        spread += v
    return RichResult(
        payload={
            "estimate": out,
            "mass": mass,
            "displacement": spread / nr,
            "method": "Barycentric projection of a plan -- Peyre & Cuturi (2019) eq. (4.19)",
        }
    )


def cheatsheet():
    return "otplan: Convert a soft transport plan to a barycentric map"


# compact alias per ledger/NAMING.md
otplantomap = ot_plan_to_map
