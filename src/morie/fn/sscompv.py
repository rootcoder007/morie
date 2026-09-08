# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validation of competing-risks model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sscompv", "survival_competing_validation"]


def survival_competing_validation(time, event_type, predicted_F):
    """
    Concordance validation of a competing-risks prediction model.

    Wolbers' concordance for the cause of interest (event_type == 1):
    a pair (i, j) is comparable when subject i has a cause-1 event and
    either T_i < T_j, or subject j experienced a competing event
    (event_type >= 2) -- competing-event subjects can never develop the
    index event, so they remain comparable at any time, exactly as in
    the Fine-Gray risk set. The pair is concordant when the predicted
    cause-1 risk of i exceeds that of j; ties count one half.

    Reference: Wolbers, Blanche, Koller, Witteman & Gerds (2014),
    "Concordance for prognostic models with competing risks",
    Biostatistics 15(3), 526-539.

    Parameters
    ----------
    time : array-like
        Observed times.
    event_type : array-like
        0 = censored, 1 = cause of interest, >= 2 = competing event.
    predicted_F : array-like
        Predicted cumulative incidence (risk) of cause 1, higher = riskier.

    Returns
    -------
    result : RichResult
        Keys: estimate (C), concordant, tied, comparable.
    """
    t = np.asarray(time, dtype=float)
    d = np.asarray(event_type, dtype=float)
    F = np.asarray(predicted_F, dtype=float)
    n = t.shape[0]
    if d.shape[0] != n or F.shape[0] != n:
        raise ValueError("time, event_type and predicted_F must have equal length")
    conc = 0.0
    tied = 0.0
    comp = 0
    for i in range(n):
        if d[i] != 1.0:
            continue
        for j in range(n):
            if j == i:
                continue
            if t[i] < t[j] or d[j] >= 2.0:
                comp += 1
                if F[i] > F[j]:
                    conc += 1.0
                elif F[i] == F[j]:
                    tied += 1.0
    if comp == 0:
        raise ValueError("no comparable pairs")
    return RichResult(payload={
        "estimate": (conc + 0.5 * tied) / comp,
        "concordant": conc,
        "tied": tied,
        "comparable": comp,
        "method": "Wolbers et al (2014) competing-risks concordance",
    })


sscompv = survival_competing_validation


def cheatsheet():
    return "sscompv(time, event_type, predicted_F) -> Wolbers competing-risks concordance."
