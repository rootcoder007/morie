# morie.fn -- function file (rootcoder007/morie)
"""Delta-shift sensitivity to a not-at-random missingness mechanism."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["missing_mechanism_sensitivity", "missingmechanismsensitivity"]


def missing_mechanism_sensitivity(Y, R, delta_grid, reference=0.0):
    """Pattern-mixture sensitivity analysis over a shift parameter delta.

    Missingness at random is not testable from the observed data: any
    value of ``E[Y | R = 0]`` is equally consistent with what was seen.
    A pattern-mixture sensitivity analysis therefore parameterises the
    departure instead of assuming it away,

        E[Y | R = 0] = E[Y | R = 1] + delta

    so the marginal mean traced over the grid is

        E[Y](delta) = P(R=1) E[Y|R=1] + P(R=0) (E[Y|R=1] + delta)
                    = E[Y|R=1] + (1 - P(R=1)) delta.

    ``delta = 0`` is the MAR answer, recovered exactly.  The mean is
    linear in delta with slope equal to the missingness rate, which is
    the useful part: the analysis is only fragile in proportion to how
    much data is absent, and with nothing missing no delta can move it.

    The TIPPING POINT is the delta at which the mean crosses
    ``reference``,

        delta* = (reference - E[Y|R=1]) / P(R=0),

    reported so the conclusion can be stated as "the missing units would
    have to differ by delta* for this to reverse" rather than as a
    single number under an untestable assumption.  With nothing missing
    there is no such delta and the field is NaN, not infinity: the
    conclusion cannot be tipped at all.

    Parameters
    ----------
    Y : array-like
        Outcomes.  Entries where ``R`` is 0 are ignored and may be any
        placeholder, including NaN.
    R : array-like of 0/1
        Response indicator, 1 where ``Y`` is observed.
    delta_grid : array-like
        Shift values to trace.
    reference : float, default 0
        Value whose crossing defines the tipping point.

    Returns
    -------
    RichResult
        ``estimate`` (mean at delta = 0), ``means`` (one per grid
        point), ``delta_grid``, ``mar_mean``, ``p_observed``,
        ``tipping_delta``, ``n_observed``, ``n``.

    References
    ----------
    Daniels, M. J. and Hogan, J. W. (2008), Missing Data in
    Longitudinal Studies: Strategies for Bayesian Modeling and
    Sensitivity Analysis, Chapman and Hall/CRC, for the pattern-mixture
    sensitivity framework and the shift parameterisation.  The book was
    not in the local corpus and could not be obtained; the mean and the
    tipping point above are elementary consequences of the shift
    equation and are stated in full so they can be checked against it.
    """
    r = C.vec(R)
    n = len(r)
    if n == 0:
        raise ValueError("R is empty")
    yraw = list(Y)
    if len(yraw) != n:
        raise ValueError("Y and R must have the same length")
    if any(v != 0.0 and v != 1.0 for v in r):
        raise ValueError("R must be binary 0/1")
    obs = [i for i in range(n) if r[i] == 1.0]
    if not obs:
        raise ValueError("no observed outcome; nothing to shift from")
    yo = [float(yraw[i]) for i in obs]
    if any(v != v for v in yo):
        raise ValueError("an outcome marked observed is NaN")
    m1 = sum(yo) / len(yo)
    p1 = len(obs) / n
    p0 = 1.0 - p1
    grid = C.vec(delta_grid)
    if len(grid) == 0:
        raise ValueError("delta_grid is empty")
    means = [m1 + p0 * d for d in grid]
    ref = float(reference)
    tip = (ref - m1) / p0 if p0 > 0.0 else float("nan")
    return RichResult(payload={
        "estimate": m1, "means": means, "delta_grid": grid,
        "mar_mean": m1, "p_observed": p1, "tipping_delta": tip,
        "n_observed": len(obs), "n": n,
        "method": "Delta-shift NMAR sensitivity (pattern mixture)"})


missingmechanismsensitivity = missing_mechanism_sensitivity


def cheatsheet():
    return "missinM: delta-shift NMAR sensitivity with a tipping point"
