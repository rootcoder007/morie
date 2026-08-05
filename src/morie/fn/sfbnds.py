# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Balke-Pearl sharp bounds on the average causal effect under an instrument."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sharp_bounds_balke_pearl"]


def _binary(v, name):
    x = [float(t) for t in np.atleast_1d(np.asarray(v, dtype=float)).tolist()]
    if len(x) == 0:
        raise ValueError("sharp_bounds_balke_pearl: %s is empty" % name)
    if any(t not in (0.0, 1.0) for t in x):
        raise ValueError("sharp_bounds_balke_pearl: %s must be binary 0/1" % name)
    return x


def sharp_bounds_balke_pearl(y, D, Z):
    r"""Sharp (tight) bounds on the average causal effect of D on Y when
    assignment Z is a valid instrument and compliance is imperfect.

    All three variables are binary.  Write

    .. math::  p_{yd.z} = P(Y = y,\; D = d \mid Z = z),

    eight numbers satisfying :math:`\sum_{y,d} p_{yd.z} = 1` for each z.
    Balke & Pearl solve the linear program over the sixteen response-type
    probabilities by enumerating the vertices of the dual constraint
    polytope, giving a closed form.  Their eq. (4), p. 1173, is the lower
    bound

    .. math::

        \mathrm{ACE} \ge \max \begin{cases}
        p_{00.0} + p_{11.1} - 1\\
        p_{00.1} + p_{11.1} - 1\\
        p_{11.0} + p_{00.1} - 1\\
        p_{00.0} + p_{11.0} - 1\\
        2p_{00.0} + p_{11.0} + p_{10.1} + p_{11.1} - 2\\
        p_{00.0} + 2p_{11.0} + p_{00.1} + p_{01.1} - 2\\
        p_{10.0} + p_{11.0} + 2p_{00.1} + p_{11.1} - 2\\
        p_{00.0} + p_{01.0} + p_{00.1} + 2p_{11.1} - 2
        \end{cases}

    and their eq. (5) the upper bound

    .. math::

        \mathrm{ACE} \le \min \begin{cases}
        1 - p_{10.0} - p_{01.1}\\
        1 - p_{01.0} - p_{10.1}\\
        1 - p_{01.0} - p_{10.0}\\
        1 - p_{01.1} - p_{10.1}\\
        2 - 2p_{01.0} - p_{10.0} - p_{10.1} - p_{11.1}\\
        2 - p_{01.0} - 2p_{10.0} - p_{00.1} - p_{01.1}\\
        2 - p_{10.0} - p_{11.0} - 2p_{01.1} - p_{10.1}\\
        2 - p_{00.0} - p_{01.0} - p_{01.1} - 2p_{10.1}
        \end{cases}

    The first four entries of each set are the Robins-Manski bounds; the last
    four are what makes the Balke-Pearl interval strictly narrower.  The
    width cannot exceed the rate of noncompliance
    :math:`P(d_1 \mid z_0) + P(d_0 \mid z_1)`, which is reported so the
    property can be checked.

    Both equations were read from a rendered image of p. 1173 of the JASA
    printing; that PDF is a scan with no text layer, so no minus sign passed
    through a text extractor.

    Parameters
    ----------
    y : array-like
        Binary outcome, one entry per unit.
    D : array-like
        Binary treatment actually received.
    Z : array-like
        Binary instrument (assignment).

    Returns
    -------
    RichResult
        ``estimate`` is the midpoint of the interval; ``lower`` and
        ``upper`` are the sharp bounds.  ``excludes_zero`` is 1 when the
        interval lies wholly above or wholly below 0, i.e. the sign of the
        effect is identified.

    References
    ----------
    Balke, A. & Pearl, J. (1997). Bounds on treatment effects from studies
    with imperfect compliance. Journal of the American Statistical
    Association 92(439), 1171-1176, eqs. (4)-(5) p. 1173.
    doi:10.1080/01621459.1997.10474074
    """
    yy = _binary(y, "y")
    dd = _binary(D, "D")
    zz = _binary(Z, "Z")
    n = len(yy)
    if len(dd) != n or len(zz) != n:
        raise ValueError("sharp_bounds_balke_pearl: y, D and Z must have the same length")

    nz = [0, 0]
    cnt = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                cnt[(a, b, c)] = 0
    for k in range(n):
        a = int(yy[k])
        b = int(dd[k])
        c = int(zz[k])
        cnt[(a, b, c)] += 1
        nz[c] += 1
    if nz[0] == 0 or nz[1] == 0:
        raise ValueError("sharp_bounds_balke_pearl: both instrument arms must be non-empty")

    def p(a, b, c):
        return cnt[(a, b, c)] / nz[c]

    p00_0, p01_0, p10_0, p11_0 = p(0, 0, 0), p(0, 1, 0), p(1, 0, 0), p(1, 1, 0)
    p00_1, p01_1, p10_1, p11_1 = p(0, 0, 1), p(0, 1, 1), p(1, 0, 1), p(1, 1, 1)

    lower_terms = [
        p00_0 + p11_1 - 1.0,
        p00_1 + p11_1 - 1.0,
        p11_0 + p00_1 - 1.0,
        p00_0 + p11_0 - 1.0,
        2.0 * p00_0 + p11_0 + p10_1 + p11_1 - 2.0,
        p00_0 + 2.0 * p11_0 + p00_1 + p01_1 - 2.0,
        p10_0 + p11_0 + 2.0 * p00_1 + p11_1 - 2.0,
        p00_0 + p01_0 + p00_1 + 2.0 * p11_1 - 2.0,
    ]
    upper_terms = [
        1.0 - p10_0 - p01_1,
        1.0 - p01_0 - p10_1,
        1.0 - p01_0 - p10_0,
        1.0 - p01_1 - p10_1,
        2.0 - 2.0 * p01_0 - p10_0 - p10_1 - p11_1,
        2.0 - p01_0 - 2.0 * p10_0 - p00_1 - p01_1,
        2.0 - p10_0 - p11_0 - 2.0 * p01_1 - p10_1,
        2.0 - p00_0 - p01_0 - p01_1 - 2.0 * p10_1,
    ]
    lo = max(lower_terms)
    up = min(upper_terms)
    # Robins-Manski: the first four entries of each set
    lo_rm = max(lower_terms[:4])
    up_rm = min(upper_terms[:4])

    noncompliance = (p01_0 + p11_0) + (p00_1 + p10_1)
    itt = (p10_1 + p11_1) - (p10_0 + p11_0)
    dz = (p01_1 + p11_1) - (p01_0 + p11_0)
    late = itt / dz if dz != 0.0 else float("nan")

    return RichResult(
        payload={
            "estimate": 0.5 * (lo + up),
            "lower": lo,
            "upper": up,
            "width": up - lo,
            "lower_manski": lo_rm,
            "upper_manski": up_rm,
            "width_manski": up_rm - lo_rm,
            "noncompliance": noncompliance,
            "itt": itt,
            "compliance_gap": dz,
            "late": late,
            "excludes_zero": 1.0 if (lo > 0.0 or up < 0.0) else 0.0,
            "p00_0": p00_0, "p01_0": p01_0, "p10_0": p10_0, "p11_0": p11_0,
            "p00_1": p00_1, "p01_1": p01_1, "p10_1": p10_1, "p11_1": p11_1,
            "n": float(n),
            "n_z0": float(nz[0]),
            "n_z1": float(nz[1]),
            "method": "Balke-Pearl sharp bounds on the ACE (Balke & Pearl 1997, eqs. 4-5)",
        }
    )


def cheatsheet():
    return "sfbnds: Balke-Pearl sharp bounds on the ATE under an instrument"


# compact alias per ledger/NAMING.md
sharpboundsbalkepearl = sharp_bounds_balke_pearl
