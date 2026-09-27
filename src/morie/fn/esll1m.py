"""L1-normalised margin of an additive classifier (ESL sec 16.2.2)."""

from ._richresult import RichResult

__all__ = ["esl_l1_margin"]


def esl_l1_margin(y, fitted, coefficients):
    r"""Margin :math:`m(f) = \min_i y_if(x_i) / \sum_k|\alpha_k|` (ESL eq 16.7).

    For :math:`f(x) = \sum_k\alpha_kT_k(x)` and labels in {-1, +1}, the
    L1-normalised margin is the smallest signed training margin divided by
    the L1 norm of the coefficients; boosting with shrinkage tends to
    maximise it (Rosset et al. 2004).

    Parameters
    ----------
    y : sequence of -1/+1
    fitted : sequence of f(x_i)
    coefficients : sequence of alpha_k

    Returns
    -------
    RichResult
        ``margin``, ``l1_norm``, ``argmin`` (0-based index of the smallest
        margin).

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 16.2.2.
    """
    yy = [float(v) for v in y]
    ff = [float(v) for v in fitted]
    if len(yy) != len(ff) or any(v not in (-1.0, 1.0) for v in yy):
        raise ValueError("need labels in {-1, +1} matching the fitted values")
    l1 = sum(abs(float(a)) for a in coefficients)
    if l1 == 0:
        raise ValueError("the coefficients are all zero")
    m = [a * b for a, b in zip(yy, ff)]
    k = min(range(len(m)), key=lambda i: m[i])
    return RichResult(
        title="L1-normalised margin",
        summary_lines=[("margin", m[k] / l1)],
        payload={"margin": m[k] / l1, "l1_norm": l1, "argmin": k},
    )


def cheatsheet():
    return "esll1m: min_i y_i f(x_i) / sum_k |alpha_k|, ESL 16.7"
