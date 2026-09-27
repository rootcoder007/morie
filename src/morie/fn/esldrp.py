"""Dirichlet posterior for category probabilities and its bootstrap link (ESL sec 8.4)."""

from ._richresult import RichResult

__all__ = ["esl_dirichlet_posterior"]


def esl_dirichlet_posterior(counts, a=0.0):
    r"""Posterior :math:`w \sim Di_L(a\mathbf 1 + N\hat w)` under a symmetric :math:`Di_L(a\mathbf 1)` prior.

    ESL eqs 8.32-8.34: :math:`N\hat w` are the category counts. With
    :math:`\alpha_\ell = a + N\hat w_\ell` and :math:`\alpha_0 = \sum_\ell\alpha_\ell`, the
    posterior mean is :math:`\alpha_\ell/\alpha_0` and the variance
    :math:`\alpha_\ell(\alpha_0-\alpha_\ell)/(\alpha_0^2(\alpha_0+1))`. With a = 0 the
    mean equals the bootstrap mean :math:`\hat w` and the variance
    :math:`\hat w_\ell(1-\hat w_\ell)/(N+1)` is close to the multinomial bootstrap
    variance :math:`\hat w_\ell(1-\hat w_\ell)/N` (eq 8.35), which is why the
    bootstrap distribution approximates a noninformative posterior.

    Parameters
    ----------
    counts : sequence of non-negative numbers
    a : float
        Prior concentration, >= 0 (0 is the noninformative limit).

    Returns
    -------
    RichResult
        ``alpha``, ``mean``, ``var``, ``bootstrap_var``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 8.4.
    """
    c = [float(v) for v in counts]
    if len(c) < 2 or min(c) < 0 or a < 0:
        raise ValueError("need at least two non-negative counts and a >= 0")
    n = sum(c)
    al = [a + v for v in c]
    a0 = sum(al)
    if a0 <= 0 or min(al) <= 0:
        raise ValueError("every posterior parameter must be positive (use a > 0 for empty categories)")
    w = [v / n for v in c]
    return RichResult(
        title="Dirichlet posterior",
        summary_lines=[("alpha0", a0)],
        payload={
            "alpha": al,
            "mean": [v / a0 for v in al],
            "var": [v * (a0 - v) / (a0 * a0 * (a0 + 1)) for v in al],
            "bootstrap_var": [v * (1 - v) / n for v in w],
        },
    )


def cheatsheet():
    return "esldrp: w ~ Di(a + counts); mean alpha / alpha0, var alpha (alpha0 - alpha) / (alpha0^2 (alpha0 + 1))"
