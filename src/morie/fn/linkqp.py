"""IRT characteristic-curve linking, Stocking-Lord method (Stocking & Lord 1983)."""

import math

from ._sci_core import minimize
from ._richresult import RichResult

__all__ = ["linkqp", "irt_linking_stocking_lord"]


def _p3pl(theta, a, b, c):
    e = math.exp(a * (theta - b))
    return c + (1.0 - c) * e / (1.0 + e)


def _quad(n=41, lo=-4.0, hi=4.0):
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def linkqp(items_from, items_to, symmetric=False, theta_points=None):
    """
    Stocking-Lord characteristic-curve linking of two IRT scales.

    Finds the constants (A, B) of theta_T = A theta_F + B by
    minimizing squared differences between TEST characteristic
    curves of the common items (plink Eq. 15; Weeks 2010): with
    transformed from-scale parameters a* = a_F/A, b* = A b_F + B,
    c* = c_F,

        F1 = (1/L1) sum_m [ sum_s P_s(theta_m)
                            - sum_s P*_s(theta_m) ]^2 .

    The non-symmetric method (default) minimizes F1 alone; the
    symmetric variant adds the reverse-direction term F2 with
    a# = A a_T, b# = (b_T - B)/A evaluated on the from scale.
    Uniform weights on an equally spaced theta grid are used.

    Sources
    -------
    Stocking, M. L. & Lord, F. M. (1983). Developing a common metric
    in item response theory. *Applied Psychological Measurement*, 7,
    201-210.
    Weeks, J. P. (2010). plink: An R package for linking
    mixed-format tests using IRT-based methods. *Journal of
    Statistical Software*, 35(12), Eqs. 8, 10, 15 (local copy
    fetched-wave3/weeks-2010-plink-JSS35.pdf).

    Parameters
    ----------
    items_from, items_to : sequences of (a, b, c) tuples
        Common-item 3PL parameters on each scale (use c = 0 for 2PL).
    symmetric : bool
        Minimize F1 + F2 instead of F1 only.
    theta_points : sequence of float, optional
        Evaluation grid (default 41 points on [-4, 4]).

    Returns
    -------
    RichResult
        Keys: A, B, criterion, symmetric, n_common.
    """
    fr = [tuple(float(v) for v in it) for it in items_from]
    to = [tuple(float(v) for v in it) for it in items_to]
    s = len(fr)
    if len(to) != s or s < 2:
        raise ValueError("need >= 2 common items with matching lengths")
    grid = list(theta_points) if theta_points is not None else _quad()
    l_norm = float(len(grid))

    def crit(x):
        A, B = x
        if A <= 0:
            return 1e10
        f1 = 0.0
        for th in grid:
            tcc_t = sum(_p3pl(th, at, bt, ct) for at, bt, ct in to)
            tcc_star = sum(_p3pl(th, af / A, A * bf + B, cf)
                           for af, bf, cf in fr)
            f1 += (tcc_t - tcc_star) ** 2
        f = f1 / l_norm
        if symmetric:
            f2 = 0.0
            for th in grid:
                tcc_f = sum(_p3pl(th, af, bf, cf) for af, bf, cf in fr)
                tcc_hash = sum(_p3pl(th, A * at, (bt - B) / A, ct)
                               for at, bt, ct in to)
                f2 += (tcc_f - tcc_hash) ** 2
            f += f2 / l_norm
        return f

    res = minimize(crit, [1.0, 0.0], method="Nelder-Mead")
    A, B = float(res.x[0]), float(res.x[1])
    return RichResult(payload={
        "A": A,
        "B": B,
        "criterion": float(res.fun),
        "symmetric": bool(symmetric),
        "n_common": s,
        "method": "Stocking-Lord characteristic-curve linking (plink Eq. 15)",
    })


# long descriptive alias (stub-era name)
irt_linking_stocking_lord = linkqp


def cheatsheet():
    return "linkqp: min sum [TCC_to - TCC_from*(A,B)]^2 (Stocking-Lord)"

# public names resolved by fn/_lazy_map.json
linking_stocking_lord = linkqp
