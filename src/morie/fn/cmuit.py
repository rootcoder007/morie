# morie.fn -- slice s03 (rootcoder007/morie)
"""Conditional mutual information.

Source consulted: Cover, T. M. and Thomas, J. A. (2006).  *Elements of
Information Theory*, 2nd ed., Wiley, section 2.5, equation (2.60):

    I(X; Y | Z) = sum_z p(z) I(X; Y | Z = z)
                = H(X | Z) + H(Y | Z) - H(X, Y | Z)

The book is not open access; the identity is quoted in its standard
published form.  The point worth keeping in view is that I(X; Y | Z) can
be *larger* than I(X; Y) -- conditioning can create dependence, as in the
XOR example -- so both are returned and the difference is the interaction
information, which is reported rather than left implicit.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .mutifo import mutual_information

__all__ = ["conditional_mutual_information"]


def conditional_mutual_information(y, x=None, y2=None, z=None):
    """I(X; Y | Z) by conditioning on each level of Z.

    Returns
    -------
    estimate : I(X; Y | Z) in nats
    mi       : the unconditional I(X; Y)
    interaction : I(X; Y) - I(X; Y | Z)
    per_level : (p(z), I(X; Y | Z = z)) per level of Z
    """
    if y2 is None:
        a = list(y)
        b = list(x)
    else:
        a = list(x)
        b = list(y2)
    c = [str(v) for v in z] if z is not None else ["0"] * len(a)
    n = len(a)
    lv = []
    for v in c:
        if v not in lv:
            lv.append(v)
    lv = sorted(lv)
    cmi = 0.0
    per = []
    for v in lv:
        idx = [i for i in range(n) if c[i] == v]
        pz = len(idx) / n
        if len(idx) < 2:
            per.append([pz, 0.0])
            continue
        sub = mutual_information([a[i] for i in idx], [b[i] for i in idx])
        cmi += pz * sub["mi"]
        per.append([pz, sub["mi"]])
    uncond = mutual_information(a, b)["mi"]
    return RichResult(
        title="Conditional mutual information",
        summary_lines=[("I(X;Y|Z)", cmi), ("I(X;Y)", uncond)],
        payload={
            "estimate": cmi,
            "cmi": cmi,
            "mi": uncond,
            "interaction": uncond - cmi,
            "per_level": per,
            "bits": cmi / math.log(2.0),
            "n": n,
            "method": "Conditional mutual information (Cover and Thomas 2006, eq. 2.60)",
        },
    )


def cheatsheet():
    return "cmuit: Conditional mutual information I(X;Y|Z)"
