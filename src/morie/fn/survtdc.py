# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time-dependent concordance index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["survtdc", "time_dep_concordance"]


def time_dep_concordance(time, event, marker, t):
    """
    Truncated time-dependent concordance index C^td(t).

    C^td(t) = P(marker_i > marker_j | T_i < T_j, T_i <= t, delta_i = 1):
    among usable pairs where subject i is observed to fail before
    subject j and no later than the horizon t, the fraction in which the
    higher-risk marker belongs to the earlier failure. Marker ties count
    one half. This is the estimator of Antolini, Boracchi & Biganzoli
    (2005), Statistics in Medicine 24(24), 3927-3944, with a scalar
    (time-fixed) marker, equal to Harrell's C truncated at t.

    Returns
    -------
    result : RichResult
        Keys: estimate (C^td), concordant, tied, comparable, t.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    marker = np.asarray(marker, dtype=float)
    t = float(t)
    n = time.shape[0]
    if event.shape[0] != n or marker.shape[0] != n:
        raise ValueError("time, event and marker must have equal length")
    conc = 0.0
    tied = 0.0
    comp = 0
    for i in range(n):
        if event[i] != 1.0 or time[i] > t:
            continue
        for j in range(n):
            if j == i or not time[i] < time[j]:
                continue
            comp += 1
            if marker[i] > marker[j]:
                conc += 1.0
            elif marker[i] == marker[j]:
                tied += 1.0
    if comp == 0:
        raise ValueError("no comparable pairs at horizon t")
    est = (conc + 0.5 * tied) / comp
    return RichResult(payload={
        "estimate": est,
        "concordant": conc,
        "tied": tied,
        "comparable": comp,
        "t": t,
        "method": "Antolini-Boracchi-Biganzoli (2005) truncated time-dependent concordance",
    })


def cheatsheet():
    return "time_dep_concordance(time, event, marker, t) -> C^td(t) truncated concordance."


survtdc = time_dep_concordance
