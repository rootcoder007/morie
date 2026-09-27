"""Generalized cross-validation for a linear smoother (ESL sec 7.10.1)."""

from ._richresult import RichResult

__all__ = ["esl_gcv"]


def esl_gcv(y, fitted, trace_S):
    r"""GCV :math:`= \frac1N\sum_i\big[(y_i - \hat f(x_i))/(1 - \mathrm{trace}(S)/N)\big]^2` (ESL eq 7.52).

    It replaces each leave-one-out factor :math:`1 - S_{ii}` of eq 7.51 by
    their average, which is what ``smooth.spline``'s GCV criterion computes.

    Parameters
    ----------
    y, fitted : sequences of floats
    trace_S : float
        Effective degrees of freedom, < N.

    Returns
    -------
    RichResult
        ``gcv``, ``rss``, ``n``.

    References
    ----------
    Craven, P. & Wahba, G. (1979). Numerische Mathematik 31, 377-403.
    """
    yy = [float(v) for v in y]
    ff = [float(v) for v in fitted]
    n = len(yy)
    if n != len(ff) or not 0 <= trace_S < n:
        raise ValueError("need y and fitted of equal length and 0 <= trace(S) < N")
    rss = sum((a - b) ** 2 for a, b in zip(yy, ff))
    g = rss / n / (1 - trace_S / n) ** 2
    return RichResult(
        title="Generalized cross-validation", summary_lines=[("gcv", g)], payload={"gcv": g, "rss": rss, "n": n}
    )


def cheatsheet():
    return "eslgcv: (RSS / N) / (1 - trace(S) / N)^2, ESL 7.52"
