# morie.fn -- function file (rootcoder007/morie)
"""Edgeworth expansions for the smoothed sign and Wilcoxon tests (Theorem 5.9)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["smthedge", "fauzi_thm5_9_edgeworth_wilcoxon"]


def smthedge(y, n, which="sign", book=False):
    r"""Edgeworth expansions for the smoothed sign and Wilcoxon tests (Theorem 5.9).

    Theorem 5.9. Under the conditions of Theorem 5.8, with a symmetric
    kernel, :math:`|f''(x)| \le M`, :math:`\int|u^4k(u)|du < \infty` and
    :math:`h_n = cn^{-d}`, :math:`\tfrac14<d<\tfrac12`:

    .. math::
        P_0\Big(\frac{\tilde S - E_0(\tilde S)}
                     {\sqrt{V_0(\tilde S)}}\le y\Big)
        &= \Phi(y) - \frac1{24n}(y^3-3y)\phi(y) + o(n^{-1}), \\
        P_0\Big(\frac{\tilde W - E_0(\tilde W)}
                     {\sqrt{V_0(\tilde W)}}\le y\Big)
        &= \Phi(y) - \frac1{20n}(y^3-3y)\phi(y) + o(n^{-1}).

    Both corrections are the same Hermite polynomial
    :math:`H_3(y)=y^3-3y`; only the constant differs, 1/24 against 1/20.
    Neither depends on :math:`F` -- that is the payoff of the whole
    section, and it holds because a symmetric fourth-order kernel is used.

    The book's printed form of the Wilcoxon line is
    :math:`\Phi(y) - (\tfrac7{20}y^3 - \tfrac{21}{20}y)\phi(y) + o(n^{-1})`,
    i.e. :math:`\tfrac7{20}H_3(y)` with NO :math:`n` in the denominator.
    That cannot be right as printed: a correction that does not shrink
    with ``n`` is not :math:`o(n^{-1})`, and it would not match the sign
    test's :math:`1/(24n)` in form. The primary source -- Maesono, Y.,
    Moriyama, T. and Lu, M. (2018), "Smoothed nonparametric tests and
    their properties", *Annals of the Institute of Statistical
    Mathematics* 70(5):969-982 (arXiv:1610.02145), Theorems 3 and 5 --
    gives :math:`\Phi(y) - \tfrac1{20n}\phi(y)H_3(y)`, which is what is
    used here. ``book=True`` reproduces the book's printed coefficients.

    ``which`` selects ``"sign"`` or ``"wilcoxon"``.

    Parameters
    ----------
    y : float or array-like
        Argument of the expansion.
    n : int
        Sample size.
    which : {"sign", "wilcoxon"}, default "sign"
        Which statistic to expand.
    book : bool, default False
        Reproduce the book's printed Wilcoxon coefficients
        ``7/20`` and ``21/20`` with no ``n``.  Has no effect for the sign
        test, where the book and the primary source agree.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``normal``, ``correction``, ``coef``, ``which``, ``book``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.9; Maesono, Moriyama and Lu (2018), AISM 70:969-982, Theorems 3 and 5.
    """
    from . import _stats_core as stats

    n = int(n)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if which not in ("sign", "wilcoxon"):
        raise ValueError('which must be "sign" or "wilcoxon".')
    yv = np.atleast_1d(np.asarray(y, dtype=float))
    base = stats.norm.cdf(yv)
    phi = stats.norm.pdf(yv)
    he3 = yv ** 3 - 3.0 * yv
    if which == "sign":
        coef = 1.0 / (24.0 * n)
        corr = coef * he3 * phi
    elif book:
        coef = 7.0 / 20.0
        corr = (7.0 / 20.0 * yv ** 3 - 21.0 / 20.0 * yv) * phi
    else:
        coef = 1.0 / (20.0 * n)
        corr = coef * he3 * phi
    return RichResult(
        payload={
            "estimate": [float(v) for v in (base - corr)],
            "normal": [float(v) for v in base],
            "correction": [float(v) for v in corr],
            "coef": float(coef),
            "which": which,
            "book": bool(book),
            "method": "Edgeworth expansion of the smoothed sign/Wilcoxon test (Theorem 5.9)",
        }
    )


fauzi_thm5_9_edgeworth_wilcoxon = smthedge


def cheatsheet():
    return "fzt59: Thm 5.9: H3 correction 1/(24n) sign, 1/(20n) Wilcoxon; the book drops the n"


# CANONICAL TEST
# >>> r = smthedge(y=0.0, n=100, which='wilcoxon')
# >>> abs(r['estimate'][0] - 0.5) < 1e-15
# True
