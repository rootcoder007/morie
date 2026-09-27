# morie.fn -- function file (rootcoder007/morie)
"""Nucleolus and prenucleolus of a transferable-utility game."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._tucore import players, sequential_nucleolus


def nucleolus(v, *, pre: bool = False, tol: float = 1e-9) -> DescriptiveResult:
    """Nucleolus (Schmeidler 1969) or prenucleolus of a TU game.

    The imputation (or, with ``pre=True``, the efficient preimputation)
    that lexicographically minimises the non-increasingly ordered vector of
    coalition excesses ``v(S) - x(S)``. Computed by the sequence of linear
    programs of Maschler, Peleg and Shapley (1979): each stage minimises
    the largest free excess, then fixes the coalitions whose excess cannot
    fall below that level, until the fixed coalitions determine ``x``. The
    first level is the least-core value epsilon (for ``pre=True``).

    :param v: Worths in binary order, length ``2**n - 1``: ``v[S - 1]`` is
        the worth of the coalition with bitmask ``S``.
    :param pre: Prenucleolus (no individual-rationality bounds).
    :param tol: Tolerance for tight and fixed excesses.
    :return: DescriptiveResult; ``value`` is the allocation, ``extra`` holds
        ``levels`` (the successive maximal excesses) and ``n_players``.

    References
    ----------
    Schmeidler, D. (1969). The nucleolus of a characteristic function game.
    SIAM Journal on Applied Mathematics 17, 1163-1170.

    Maschler, M., Peleg, B. and Shapley, L. S. (1979). Geometric properties
    of the kernel, nucleolus, and related solution concepts. Mathematics of
    Operations Research 4, 303-338.

    Examples
    --------
    >>> r = nucleolus([0, 0, 60, 0, 60, 60, 72])
    >>> [round(a, 9) for a in r.value]
    [24.0, 24.0, 24.0]
    >>> [round(a, 9) for a in r.extra["levels"]]
    [12.0]
    """
    n = players(v)
    x, levels = sequential_nucleolus(v, pre, tol)
    return DescriptiveResult(
        name="prenucleolus" if pre else "nucleolus",
        value=x,
        extra={"levels": levels, "n_players": n},
    )


tunucl = nucleolus


def cheatsheet() -> str:
    return "nucleolus(v) -> nucleolus / prenucleolus of a TU game (sequential LPs)"
