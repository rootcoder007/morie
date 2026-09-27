# morie.fn -- function file (rootcoder007/morie)
"""Kim-Fording median voter from party positions and vote shares."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._qpcore import ssum


def kim_fording_median(
    positions, voteshares, *, adjusted: bool = False, scalemin: float = -100.0, scalemax: float = 100.0
) -> DescriptiveResult:
    """Kim and Fording (1998) median voter position from party positions and votes.

    Parties are ordered on the scale; each party's voters are spread
    uniformly between the midpoints to its neighbours (the scale ends for
    the outermost parties, or, with ``adjusted``, the reflections of the
    neighbouring midpoints; Kim and Fording 2003). The median voter lies in
    the interval where the cumulative vote share crosses one half:
    ``lower + (0.5 - cumulative before) / share * (upper - lower)``. Parties
    at the same position are merged. This is
    ``manifestoR::median_voter``.

    :param positions: Party positions.
    :param voteshares: Party vote shares (any scale; normalised).
    :param adjusted: Use the adjusted outer bounds.
    :param scalemin: Lower end of the scale.
    :param scalemax: Upper end of the scale.
    :return: DescriptiveResult; ``value`` is the median voter position.

    References
    ----------
    Kim, H. and Fording, R. C. (1998). Voter ideology in Western
    democracies, 1946-1989. European Journal of Political Research 33,
    73-97.

    Kim, H. and Fording, R. C. (2003). Voter ideology in Western
    democracies: an update. European Journal of Political Research 42,
    95-105.

    Examples
    --------
    >>> kim_fording_median([-20, 5, 30], [30, 25, 45]).value
    12.5
    """
    agg = {}
    for x, v in zip(positions, voteshares):
        agg[float(x)] = agg.get(float(x), 0.0) + float(v)
    xs = sorted(agg)
    tot = ssum(agg.values())
    sh = [agg[x] / tot for x in xs]
    m = len(xs)
    if adjusted and m >= 2:
        lo0, hi0 = 2 * xs[0] - xs[1], 2 * xs[-1] - xs[-2]
        left = [(lo0 + xs[0]) / 2] + [(xs[k - 1] + xs[k]) / 2 for k in range(1, m)]
        right = [(xs[k] + xs[k + 1]) / 2 for k in range(m - 1)] + [(xs[-1] + hi0) / 2]
    else:
        left = [scalemin] + [(xs[k - 1] + xs[k]) / 2 for k in range(1, m)]
        right = [(xs[k] + xs[k + 1]) / 2 for k in range(m - 1)] + [scalemax]
    cum = 0.0
    for k in range(m):
        before = cum
        cum += sh[k]
        if cum >= 0.5:
            med = left[k] + (0.5 - before) / sh[k] * (right[k] - left[k])
            return DescriptiveResult(name="kim_fording_median", value=med, extra={"positions": xs, "shares": sh})
    raise ValueError("vote shares do not accumulate to one half")


medvot = kim_fording_median


def cheatsheet() -> str:
    return "kim_fording_median(positions, voteshares) -> Kim-Fording median voter position"
