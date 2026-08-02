# morie.fn -- function file (rootcoder007/morie)
"""Lemma 3.1: asymptotic representation of kernel quantile estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_lemma_3_1", "fauzi_lem3_1_asymp_rep"]


def fauzi_lemma_3_1(x, p, h=None, q_true=None):
    r"""Lemma 3.1 (Fauzi Ch. 3): the asymptotic representation of the
    kernel quantile estimator as an average plus a negligible
    remainder,

    .. math:: \hat Q_{p,h} - Q(p)
              = \frac{1}{n}\sum_{i=1}^{n}
                \frac{p - \mathbf 1\{X_i \le Q(p)\}}
                     {f(Q(p))} + R_n ,

    with :math:`R_n` of smaller order.

    A Bahadur-type representation, and it is the workhorse of the
    chapter: once the estimator is an i.i.d. AVERAGE plus a
    remainder, its limiting normality, its variance
    :math:`p(1-p)/(nf^2)` and the Edgeworth expansion that refines
    them all follow from standard theory for sums. Without it each
    would need its own argument.

    The influence function :math:`(p - \mathbf 1\{X \le Q\})/f(Q)`
    is returned, since it is what every subsequent variance and
    expansion is built from.

    Parameters
    ----------
    x : array-like
        Sample.
    p : float
        Probability level in (0, 1).
    h : float, optional
        Bandwidth for the kernel quantile estimate.

    Returns
    -------
    RichResult
        keys: ``influence``, ``linear_term``, ``estimate``,
        ``remainder``, ``density_at_quantile``, ``centre``, ``centred_at``,
        ``asymptotic_variance``,
        ``representation``, ``n``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Lemma 3.1. From the PDF.
    """
    from ._fauzi import kernel_K
    from .fzkqe import fauzi_kernel_quantile

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    pp = float(p)
    if not 0 < pp < 1:
        raise ValueError(f"p must lie strictly in (0, 1), got {pp}.")
    # Lemma 3.1 expands Q_hat - Q(p) about the POPULATION quantile.
    # Centring on the sample quantile instead makes the linear term
    # (p - F_n(Q))/f vanish identically -- F_n at its own p-quantile
    # is p up to 1/n -- so the "representation" would be a remainder
    # and nothing else, and the asymptotic variance it licenses would
    # be unsupported. When the truth is unknown the sample quantile is
    # still the only available centre, so it is used and flagged
    # rather than silently substituted.
    if q_true is None:
        centred_at = ("sample quantile -- the linear term is degenerate here; "
                      "supply q_true for the lemma as stated")
        Q = float(np.quantile(xv, pp))
    else:
        centred_at = "population quantile (supplied)"
        Q = float(q_true)
    hb = 1.06 * float(np.std(xv, ddof=1)) * n ** -0.2
    fQ = float(np.mean(kernel_K((Q - xv) / hb)) / hb)
    if fQ <= 0:
        raise ValueError("the estimated density at the quantile is zero; "
                         "the representation divides by it.")
    infl = (pp - (xv <= Q).astype(float)) / fQ
    lin = float(infl.mean())
    est = float(fauzi_kernel_quantile(xv, pp, h=h)["quantile"][0])
    return RichResult(payload={
        "influence": infl, "linear_term": lin, "estimate": est,
        "remainder": float(est - Q - lin),
        "density_at_quantile": fQ,
        "centre": Q, "centred_at": centred_at,
        "asymptotic_variance": float(pp * (1 - pp) / (n * fQ ** 2)),
        "representation": "Bahadur-type: an i.i.d. average plus a smaller-order "
                          "remainder, which is what makes normality, the "
                          "variance and the Edgeworth expansion all follow "
                          "from standard theory for sums",
        "n": int(n),
        "method": "Lemma 3.1: asymptotic representation of the kernel quantile estimator"})


def cheatsheet():
    return "fzl31: once it is an average plus a remainder, everything else follows from sums"


#: Catalogue alias for :func:`fauzi_lemma_3_1`.
fauzi_lem3_1_asymp_rep = fauzi_lemma_3_1
