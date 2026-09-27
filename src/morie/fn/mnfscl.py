# morie.fn -- function file (rootcoder007/morie)
"""Left-right scales from Manifesto Project category percentages."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import ssum

RILE_RIGHT = (104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606)
RILE_LEFT = (103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701)


def _get(per, code):
    for k in (code, str(code), f"per{code}"):
        if k in per:
            v = per[k]
            return 0.0 if v is None or v != v else float(v)
    return 0.0


def manifesto_scales(per, total=None, *, pos=RILE_RIGHT, neg=RILE_LEFT, zero_offset: float = 0.5) -> DescriptiveResult:
    """Left-right scales from Manifesto Project category percentages.

    With ``R`` and ``L`` the summed percentages of the ``pos`` and ``neg``
    categories (by default the right and left RILE categories of Laver and
    Budge 1992): ``rile = R - L``; the Kim and Fording (1998) ratio scale
    ``(R - L) / (R + L)``; the plain ratio ``R / L``; and, when the number
    of quasi-sentences ``total`` is given, the logit scale of Lowe et al.
    (2011) ``log((R total / 100 + 0.5) / (L total / 100 + 0.5))``. These are
    ``manifestoR::rile``, ``scale_ratio_1``, ``scale_ratio_2`` and
    ``logit_rile``.

    :param per: Mapping from category (``104``, ``"104"`` or ``"per104"``)
        to its percentage, or a list of such mappings (one per document).
    :param total: Quasi-sentence count(s) for the logit scale.
    :param pos: Categories scored positive.
    :param neg: Categories scored negative.
    :param zero_offset: Added to each count in the logit scale.
    :return: DescriptiveResult; ``value`` is the RILE score(s); ``extra``
        has ``rile``, ``ratio_1``, ``ratio_2``, ``logit``.

    References
    ----------
    Laver, M. and Budge, I. (1992). Party Policy and Government Coalitions.
    Macmillan.

    Kim, H. and Fording, R. C. (1998). Voter ideology in Western
    democracies, 1946-1989. European Journal of Political Research 33,
    73-97.

    Lowe, W., Benoit, K., Mikhaylov, S. and Laver, M. (2011). Scaling
    policy preferences from coded political texts. Legislative Studies
    Quarterly 36, 123-155.

    Examples
    --------
    >>> r = manifesto_scales({"per104": 5.0, "per401": 7.5, "per403": 2.0, "per504": 10.0}, total=200)
    >>> r.value, round(r.extra["logit"][0], 12)
    ([0.5], 0.040005334614)
    """
    docs = per if isinstance(per, list) else [per]
    tot = (total if isinstance(total, list) else [total]) if total is not None else [None] * len(docs)
    rile, r1, r2, lg = [], [], [], []
    for d, t in zip(docs, tot):
        R = ssum(_get(d, c) for c in pos)
        L = ssum(_get(d, c) for c in neg)
        rile.append(R - L)
        r1.append((R - L) / (R + L) if R + L != 0 else float("nan"))
        r2.append(R / L if L != 0 else float("nan"))
        lg.append(
            math.log((R * t / 100 + zero_offset) / (L * t / 100 + zero_offset)) if t is not None else float("nan")
        )
    return DescriptiveResult(
        name="manifesto_scales", value=rile, extra={"rile": rile, "ratio_1": r1, "ratio_2": r2, "logit": lg}
    )


mnfscl = manifesto_scales


def cheatsheet() -> str:
    return "manifesto_scales(per) -> RILE, Kim-Fording ratio and Lowe logit scales"
