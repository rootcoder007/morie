# morie.fn -- function file (rootcoder007/morie)
"""Hill estimator of the tail index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_hill", "evt_hill_estimator"]


def ev_hill(x, k=None):
    r"""The Hill (1975) estimator of the extreme-value index of a
    heavy-tailed distribution,

    .. math:: \hat\xi_H = \frac1k \sum_{i=1}^{k}
              \log X_{(n-i+1)} - \log X_{(n-k)},

    the mean log-excess of the top :math:`k` order statistics over
    the :math:`k+1`-st.

    Three things the formula does not say but the estimator's use
    depends on, all reflected here. It is only consistent for
    :math:`\xi > 0` -- Frechet-type, regularly varying tails -- and
    returns nonsense for bounded or exponential tails, so
    non-positive data at the threshold are an error rather than a
    silent log of a negative number. The choice of :math:`k` is THE
    problem: variance falls and bias grows with :math:`k`, so when
    ``k`` is omitted the whole Hill "plot" over a range of k is
    returned alongside the default, because a single number hides
    exactly the instability a user needs to see. And the asymptotic
    standard error :math:`\hat\xi/\sqrt k` is only honest in the
    bias-free regime -- it is reported with that caveat attached.

    ``tail_alpha`` is :math:`1/\hat\xi`, the Pareto exponent, since
    half the literature parameterises that way.

    Parameters
    ----------
    x : array-like
        Sample; the top order statistics must be positive.
    k : int, optional
        Number of top order statistics. When omitted,
        ``sqrt(n)`` is used and the full plot is returned.

    Returns
    -------
    RichResult
        keys: ``xi``, ``tail_alpha``, ``se``, ``k``, ``threshold``,
        ``hill_plot_k``, ``hill_plot_xi`` (when k was omitted),
        ``valid_for``, ``n``, ``method``.

    References
    ----------
    Hill, B. M. (1975), "A simple general approach to inference
    about the tail of a distribution", *Annals of Statistics*
    3:1163-1174.
    """
    from ._evt import top_order

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    auto = k is None
    kk = int(np.sqrt(n)) if auto else int(k)
    top = top_order(xv, kk)
    if top[-1] <= 0:
        raise ValueError(
            "the threshold order statistic is not positive; the Hill "
            "estimator is only defined for positive heavy-tailed data "
            "(xi > 0), and log of a non-positive value is not a tail index.")
    logs = np.log(top)
    xi = float(np.mean(logs[:-1]) - logs[-1])
    payload = {
        "xi": xi, "tail_alpha": (1.0 / xi if xi > 0 else np.inf),
        "se": xi / np.sqrt(kk),
        "se_caveat": "xi/sqrt(k) is the bias-free asymptotic SE; in the "
                     "biased regime (k too large) it understates the error",
        "k": kk, "threshold": float(top[-1]),
        "valid_for": "xi > 0 only -- Frechet-type, regularly varying tails; "
                     "for xi of any sign use ev_pickands or ev_dedh",
        "n": int(n),
        "method": "Hill (1975): mean log-excess of the top k order statistics"}
    if auto:
        ks = np.arange(2, min(n // 2, 500))
        xs_sorted = np.sort(xv)[::-1]
        lx = np.log(np.maximum(xs_sorted, 1e-300))
        cums = np.cumsum(lx)
        plot = cums[ks - 1] / ks - lx[ks]
        payload["hill_plot_k"] = ks
        payload["hill_plot_xi"] = plot
        payload["k_choice_note"] = ("variance falls and bias grows with k; "
                                    "the plot is returned because a single "
                                    "number hides the instability")
    return RichResult(payload=payload)


def cheatsheet():
    return "evhill: consistent only for xi > 0, and k is THE problem -- look at the plot"


#: Catalogue alias for :func:`ev_hill`.
evt_hill_estimator = ev_hill


# compact alias per ledger/NAMING.md
evhill = ev_hill
