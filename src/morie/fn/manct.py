# morie.fn -- slice s03 (rootcoder007/morie)
"""Continuity correction for a 2x2 meta-analytic table.

Source consulted: Sweeting, M. J., Sutton, A. J. and Lambert, P. C.
(2004).  What to add to nothing?  Use and avoidance of continuity
corrections in meta-analysis of sparse data.  *Statistics in Medicine*
23(9), 1351-1375.  They compare three schemes for a table with a zero
cell: the constant correction (add the same c to every cell), the
treatment-arm correction (add a value proportional to the reciprocal of
the opposite arm size), and the empirical correction.  The paper is
paywalled; the constant scheme implemented here is the one the module's
own formula line specifies,

    a* = a + c,  b* = b + c,  c* = c_cell + c,  d* = d + c

and it is stated identically wherever the correction is defined.

By default the correction is applied only when at least one cell is
zero, which is the convention Sweeting et al. describe as standard
practice; pass ``always=True`` to add it unconditionally.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["ma_continuity_correction"]


def ma_continuity_correction(a, b, c, d, cc=0.5, always=False):
    """Add a constant continuity correction to a 2x2 table.

    Parameters
    ----------
    a, b : float
        Events and non-events in the treatment arm.
    c, d : float
        Events and non-events in the control arm.
    cc : float
        The constant to add; 0.5 is the usual choice.
    always : bool
        Add the correction even when no cell is zero.

    Returns
    -------
    RichResult with payload:
        a_adj, b_adj, c_adj, d_adj : corrected cells
        estimate : log odds ratio of the corrected table
        se       : its standard error, sqrt(sum of reciprocals)
        applied  : whether the correction was actually added
    """
    a0, b0, c0, d0 = float(a), float(b), float(c), float(d)
    zero = (a0 == 0.0) or (b0 == 0.0) or (c0 == 0.0) or (d0 == 0.0)
    add = float(cc) if (always or zero) else 0.0
    aa, bb, ccc, dd = a0 + add, b0 + add, c0 + add, d0 + add
    if aa > 0.0 and bb > 0.0 and ccc > 0.0 and dd > 0.0:
        lor = math.log((aa * dd) / (bb * ccc))
        se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / ccc + 1.0 / dd)
    else:
        lor = float("nan")
        se = float("nan")
    return RichResult(
        title="Continuity correction for a 2x2 table",
        summary_lines=[("log OR", lor), ("SE", se)],
        payload={
            "a_adj": aa,
            "b_adj": bb,
            "c_adj": ccc,
            "d_adj": dd,
            "estimate": lor,
            "se": se,
            "applied": add > 0.0,
            "cc": add,
            "method": "Constant continuity correction for sparse 2x2 tables",
        },
    )


def cheatsheet():
    return "manct: Add continuity correction c to zero cells"
