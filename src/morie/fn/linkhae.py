"""IRT characteristic-curve linking, Haebara method (Haebara 1980)."""

import math

from ._sci_core import minimize
from ._richresult import RichResult

__all__ = ["linkhae", "irt_linking_haebara"]


def _p3pl(theta, a, b, c):
    e = math.exp(a * (theta - b))
    return c + (1.0 - c) * e / (1.0 + e)


def _quad(n=41, lo=-4.0, hi=4.0):
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def linkhae(items_from, items_to, symmetric=False, theta_points=None):
    """
    Haebara characteristic-curve linking of two IRT scales.

    Finds the constants (A, B) of theta_T = A theta_F + B by
    minimizing the summed squared differences between ITEM
    characteristic curves of the common items (plink Eqs. 14a-14c;
    Weeks 2010): with to-scale probabilities P_s(theta_m) and
    transformed from-scale probabilities P*_s computed with
    a* = a_F/A, b* = A b_F + B, c* = c_F,

        Q1 = (1/L1) sum_m sum_s [P_s(theta_m) - P*_s(theta_m)]^2.

    The non-symmetric method (default) minimizes Q1 alone; the
    symmetric method (Haebara's own proposal) minimizes Q1 + Q2,
    where Q2 evaluates the reverse transformation a# = A a_T,
    b# = (b_T - B)/A on the from scale.  Uniform weights on an
    equally spaced theta grid are used.

    Sources
    -------
    Haebara, T. (1980). Equating logistic ability scales by a
    weighted least squares method. *Japanese Psychological
    Research*, 22, 144-149.
    Weeks, J. P. (2010). plink: An R package for linking
    mixed-format tests using IRT-based methods. *Journal of
    Statistical Software*, 35(12), Eqs. 8, 10, 14 (local copy
    fetched-wave3/weeks-2010-plink-JSS35.pdf).

    Parameters
    ----------
    items_from, items_to : sequences of (a, b, c) tuples
        Common-item 3PL parameters on each scale (use c = 0 for 2PL).
    symmetric : bool
        Minimize Q1 + Q2 instead of Q1 only.
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
    m = len(grid)
    l_norm = float(m * s)

    def crit(x):
        A, B = x
        if A <= 0:
            return 1e10
        q1 = 0.0
        for th in grid:
            for (af, bf, cf), (at, bt, ct) in zip(fr, to):
                p_t = _p3pl(th, at, bt, ct)
                p_star = _p3pl(th, af / A, A * bf + B, cf)
                q1 += (p_t - p_star) ** 2
        q = q1 / l_norm
        if symmetric:
            q2 = 0.0
            for th in grid:
                for (af, bf, cf), (at, bt, ct) in zip(fr, to):
                    p_f = _p3pl(th, af, bf, cf)
                    p_hash = _p3pl(th, A * at, (bt - B) / A, ct)
                    q2 += (p_f - p_hash) ** 2
            q += q2 / l_norm
        return q

    res = minimize(crit, [1.0, 0.0], method="Nelder-Mead")
    A, B = float(res.x[0]), float(res.x[1])
    return RichResult(payload={
        "A": A,
        "B": B,
        "criterion": float(res.fun),
        "symmetric": bool(symmetric),
        "n_common": s,
        "method": "Haebara characteristic-curve linking (plink Eq. 14)",
    })


# long descriptive alias (stub-era name)
irt_linking_haebara = linkhae


def cheatsheet():
    return "linkhae: min sum [P_to - P_from*(A,B)]^2 over item curves"

# public names resolved by fn/_lazy_map.json
linking_haebara = linkhae
linkinghaebara = linkhae
