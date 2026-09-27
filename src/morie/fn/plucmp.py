# morie.fn -- function file (rootcoder007/morie)
"""Hotelling-Downs plurality competition on a line: shares, best responses, equilibrium."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._qpcore import ssum


def _shares(voters, weights, pos):
    total = ssum(weights)
    out = [0.0] * len(pos)
    for v, w in zip(voters, weights):
        d = [abs(v - c) for c in pos]
        best = min(d)
        near = [k for k, t in enumerate(d) if t - best <= 1e-12 * max(1.0, abs(v))]
        for k in near:
            out[k] += w / len(near)
    return [t / total for t in out]


def _median_interval(voters, weights):
    pairs = sorted(zip(voters, weights))
    total = ssum(weights)
    acc, lo, hi = 0.0, None, None
    for v, w in pairs:
        before = acc
        acc += w
        if lo is None and acc >= total / 2 - 1e-12 * total:
            lo = v
        if hi is None and before >= total / 2 - 1e-12 * total and before > 0:
            hi = v
        if lo is not None and acc > total / 2 + 1e-12 * total:
            hi = v if hi is None else hi
            break
    return lo, hi if hi is not None else lo


def plurality_competition(voters, positions=None, *, weights=None) -> DescriptiveResult:
    """Vote shares, best responses and Nash test for plurality competition on a line.

    Each voter supports the nearest candidate (ties split equally);
    candidates maximise their vote share (Hotelling 1929, Downs 1957).
    With a finite electorate a candidate's share, as its position ``y``
    varies, changes only where it becomes equidistant with a rival for some
    voter, i.e. at ``y = c_k`` or ``y = 2 v - c_k``; the best response is
    found exactly by evaluating those breakpoints and the midpoints between
    them. The configuration is a Nash equilibrium when no candidate can
    raise its share. Without ``positions`` the two-candidate equilibrium is
    returned: both candidates at the median (any common point of the median
    interval).

    :param voters: Voter ideal points (numbers).
    :param positions: Candidate positions; ``None`` for the two-candidate
        Downsian equilibrium.
    :param weights: Voter weights (default 1).
    :return: DescriptiveResult; ``value`` is the vote shares; ``extra`` has
        ``positions``, ``best_response`` (location and share for each
        candidate), ``gain`` (best-response share minus current share) and
        ``is_equilibrium``, plus ``median_interval``.

    References
    ----------
    Hotelling, H. (1929). Stability in competition. Economic Journal 39, 41-57.

    Downs, A. (1957). An Economic Theory of Democracy. Harper and Row.

    Eaton, B. C. and Lipsey, R. G. (1975). The principle of minimum
    differentiation reconsidered. Review of Economic Studies 42, 27-49.

    Examples
    --------
    >>> r = plurality_competition([0.0, 1.0, 2.0, 7.0, 9.0])
    >>> r.extra["positions"], r.value, r.extra["is_equilibrium"]
    ([2.0, 2.0], [0.5, 0.5], True)
    >>> plurality_competition([0.0, 1.0, 2.0, 7.0, 9.0], [1.0, 2.0]).extra["gain"]
    [0.09999999999999998, 0.0]
    """
    v = [float(t) for t in voters]
    w = [1.0] * len(v) if weights is None else [float(t) for t in weights]
    med = _median_interval(v, w)
    pos = [med[0], med[0]] if positions is None else [float(t) for t in positions]
    shares = _shares(v, w, pos)
    best, gain = [], []
    for k in range(len(pos)):
        others = pos[:k] + pos[k + 1 :]
        pts = sorted(set(others + [2 * x - c for x in v for c in others]))
        cand = list(pts) + [(a + b) / 2 for a, b in zip(pts, pts[1:])]
        cand += [pts[0] - 1.0, pts[-1] + 1.0] if pts else [pos[k]]
        top_y, top_s = pos[k], shares[k]
        for y in cand:
            s = _shares(v, w, others[:k] + [y] + others[k:])[k]
            if s > top_s + 1e-12:
                top_y, top_s = y, s
        best.append([top_y, top_s])
        gain.append(top_s - shares[k])
    return DescriptiveResult(
        name="plurality_competition",
        value=shares,
        extra={
            "positions": pos,
            "best_response": best,
            "gain": gain,
            "is_equilibrium": all(g <= 1e-12 for g in gain),
            "median_interval": list(med),
        },
    )


plucmp = plurality_competition


def cheatsheet() -> str:
    return "plurality_competition(voters, positions) -> Hotelling-Downs shares, best responses, Nash test"
