# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P17: incapacitation (``research/lean/P17Incapacitation.lean``; Avi-Itzhak & Shinnar 1973).

* ``Research.P17.steady_state_rate`` / ``cycle_rate`` / ``rate_le_lam``
* ``Research.P17.prevented_share_eq`` / ``prevented_share_lt_one``
* ``Research.P17.rate_antitone_in_S`` / ``rate_antitone_in_q``
* ``Research.P17.marginal_prevention_eq`` / ``marginal_prevention_pos`` / ``high_rate_more_prevented``

R parity: ``rmorie`` ``R/incapacitation.R`` (``morie_incapacitation``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = ["incapacitation", "incapacitation_career"]


def incapacitation(lam, q, S, shares=None):
    """Crime rate, prevented share and the marginal sentence year under the incapacitation model.

    Examples
    --------
    >>> r = incapacitation(lam=[2, 10], q=0.1, S=1, shares=[0.8, 0.2])
    >>> ([round(float(v), 12) for v in r["rate"]], {k: round(v, 12) for k, v in r.attrs["aggregate"].items()})
    ([1.666666666667, 5.0], {'free_rate': 3.6, 'incapacitated_rate': 2.333333333333, 'prevented_share': 0.351851851852})
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    qq = np.atleast_1d(np.asarray(q, dtype=float))
    SS = np.atleast_1d(np.asarray(S, dtype=float))
    n = max(lam.shape[0], qq.shape[0], SS.shape[0])
    rec = lambda a: np.array([float(a[i % a.shape[0]]) for i in range(n)])  # noqa: E731
    lam, qq, SS = rec(lam), rec(qq), rec(SS)
    if np.any(lam <= 0) or np.any((qq <= 0) | (qq > 1)) or np.any(SS < 0):
        raise ValueError("lambda must be positive, q in (0, 1] and S non-negative")
    rate = lam / (1 + lam * qq * SS)
    prevented = lam * qq * SS / (1 + lam * qq * SS)
    marginal = lam**2 * qq / ((1 + lam * qq * SS) * (1 + lam * qq * (SS + 1)))
    out = pd.DataFrame(
        {"lambda": lam, "q": qq, "S": SS, "rate": rate, "prevented_share": prevented, "marginal_prevention": marginal}
    )
    if shares is not None:
        sh = np.atleast_1d(np.asarray(shares, dtype=float))
        if sh.shape[0] != n or np.any(sh < 0) or abs(float(sh.sum()) - 1) > 1e-10:
            raise ValueError("shares must be one non-negative number per group, summing to one")
        free = float(np.sum(sh * lam))
        inc = float(np.sum(sh * rate))
        out.attrs["aggregate"] = {"free_rate": free, "incapacitated_rate": inc, "prevented_share": 1 - inc / free}
    out.attrs["theorems"] = [
        "Research.P17.steady_state_rate",
        "Research.P17.cycle_rate",
        "Research.P17.prevented_share_eq",
        "Research.P17.prevented_share_lt_one",
        "Research.P17.rate_antitone_in_S",
        "Research.P17.rate_antitone_in_q",
        "Research.P17.marginal_prevention_eq",
        "Research.P17.high_rate_more_prevented",
    ]
    return out


def incapacitation_career(lambda_path, t0, S, replacement=0.0) -> dict:
    """Incapacitation under desistance (a non-increasing rate path) and replacement.

    ``Research.P17Replacement.prevented_le_const`` / ``prevented_ge_const`` / ``later_sentence_prevents_less`` /
    ``replaced_antitone`` / ``prevented_net_le``.

    Examples
    --------
    >>> r = incapacitation_career([12, 10, 8, 6, 5, 4, 3, 2, 2, 1], t0=2, S=3, replacement=0.25)
    >>> (r["prevented"], r["prevented_net"], r["upper"], r["lower"], r["later"])
    (19.0, 14.25, 24.0, 12.0, 15.0)
    """
    try:
        lam = [float(x) for x in lambda_path]
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda_path must be non-negative numbers") from exc
    if any(math.isnan(x) or x < 0 for x in lam):
        raise ValueError("lambda_path must be non-negative numbers")
    if any(b > a for a, b in zip(lam, lam[1:])):
        raise ValueError("lambda_path must be non-increasing (desistance)")
    if t0 < 0 or t0 != int(t0):
        raise ValueError("t0 must be a non-negative integer")
    if S < 1 or int(S) != S:
        raise ValueError("S must be a positive integer")
    if replacement < 0 or replacement > 1:
        raise ValueError("replacement must lie in [0, 1]")
    t0, S = int(t0), int(S)
    if len(lam) < t0 + S + 1:
        raise ValueError("lambda_path must have length at least t0 + S + 1")
    prevented = math.fsum(lam[t0 : t0 + S])
    later = math.fsum(lam[t0 + 1 : t0 + 1 + S]) if len(lam) >= t0 + S + 2 else math.nan
    return {
        "prevented": prevented,
        "prevented_net": (1 - replacement) * prevented,
        "upper": S * lam[t0],
        "lower": S * lam[t0 + S],
        "later": later,
        "replacement": replacement,
        "factor": 1 - replacement,
        "theorems": [
            "Research.P17Replacement.prevented_le_const",
            "Research.P17Replacement.prevented_ge_const",
            "Research.P17Replacement.later_sentence_prevents_less",
            "Research.P17Replacement.replaced_antitone",
            "Research.P17Replacement.prevented_net_le",
        ],
    }
