"""One-dimensional agenda-setter equilibrium (Romer and Rosenthal 1978).

Romer, T. and Rosenthal, H. (1978). Political resource allocation, controlled agendas, and the
status quo. Public Choice 33(4), 27-43.
"""

from ._richresult import RichResult

__all__ = ["agenda_setter_equilibrium"]


def agenda_setter_equilibrium(setter, median, status_quo, options=None):
    r"""Take-it-or-leave-it proposal of a monopoly agenda setter facing a median voter.

    With symmetric single-peaked preferences the median voter accepts x iff
    |x - m| <= |q - m|, the acceptance set [m - |q - m|, m + |q - m|]. The setter proposes the
    acceptable point closest to its ideal s (the clamp of s to that interval), or among
    ``options`` the acceptable option closest to s, keeping q when none is acceptable. Setter
    power is the distance the outcome moves from the status quo.

    Parameters
    ----------
    setter, median, status_quo : float
    options : sequence, optional
        Discrete feasible proposals.

    Returns
    -------
    RichResult
        Keys: outcome, acceptance_set, power, setter_gain (|q - s| - |outcome - s|).

    References
    ----------
    Romer, T. and Rosenthal, H. (1978). Public Choice 33(4), 27-43.

    Examples
    --------
    >>> agenda_setter_equilibrium(10.0, 4.0, 1.0)["outcome"]
    7.0
    """
    s, m, q = float(setter), float(median), float(status_quo)
    lo, hi = m - abs(q - m), m + abs(q - m)
    if options is None:
        out = min(max(s, lo), hi)
    else:
        acc = [float(o) for o in options if lo <= float(o) <= hi]
        out = min(acc, key=lambda o: (abs(o - s), o)) if acc else q
    return RichResult(
        title="Agenda-setter equilibrium",
        summary_lines=[("outcome", out)],
        payload={
            "outcome": out,
            "acceptance_set": (lo, hi),
            "power": abs(out - q),
            "setter_gain": abs(q - s) - abs(out - s),
        },
    )


def cheatsheet():
    return "svsetr: Romer-Rosenthal monopoly agenda-setter equilibrium in one dimension"
