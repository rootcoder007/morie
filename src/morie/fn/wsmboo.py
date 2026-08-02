# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap variance estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_bootstrap"]


def wasserman_bootstrap(data, T, B=1000, seed=0, ddof=1):
    r"""Bootstrap estimate of the variance of a statistic, ESL
    Eq. (7.53):

    .. math:: \widehat{\mathrm{Var}}[S(\mathbf Z)] = \frac1{B-1}
              \sum_{b=1}^B \left(S(\mathbf Z^{*b}) - \bar S^*\right)^2,
              \qquad \bar S^* = \frac1B\sum_b S(\mathbf Z^{*b}).

    Note the denominator: :math:`B - 1`, not :math:`B`. ESL (7.53)
    prints :math:`B-1`, and it is the same reason a sample variance
    carries :math:`n-1` -- the replicates are centred at their own
    mean, which costs a degree of freedom. With the ``B = 100`` the
    book suggests the two differ by 1%, which is small; with the
    ``B = 20`` someone in a hurry will use it is 5%. Both are
    available through ``ddof`` and the alternative is always
    reported.

    The book's framing is worth keeping: this is a Monte-Carlo
    estimate of the variance of :math:`S(\mathbf Z)` under sampling
    from the EMPIRICAL distribution :math:`\hat F`. It is therefore
    exact only for the bootstrap world, and its relevance to the real
    one rests on :math:`\hat F` being close to :math:`F` in a sense
    strong enough for the statistic at hand -- Kosorok's Ch. 10
    result, that the bootstrap is consistent for Donsker classes.

    Parameters
    ----------
    data : array-like
        Sample; rows are observations.
    T : callable
        The statistic :math:`S`, applied to a sample.
    B : int, default 1000
        Replicates.
    seed : int, default 0
        Resampling seed.
    ddof : int, default 1
        Divisor is ``B - ddof``. The book's (7.53) is ``ddof = 1``.

    Returns
    -------
    RichResult
        keys: ``value`` (the variance), ``se``, ``variance_ddof1``,
        ``variance_ddof0``, ``mean_replicate``, ``bias``,
        ``replicates``, ``B``, ``n``, ``ddof``, ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Eq. (7.53) and Fig. 7.12.
    Read from the PDF. Efron, B. (1979), "Bootstrap methods: another
    look at the jackknife", *Annals of Statistics* 7:1-26. Kosorok
    (2008), Ch. 10, for consistency over Donsker classes.
    """
    from ._wsm import bootstrap_replicates

    d = np.asarray(data, dtype=float)
    n = d.shape[0]
    if not callable(T):
        raise ValueError("T must be callable on a sample.")
    reps = bootstrap_replicates(d, T, B=B, seed=seed)
    Bn = reps.size
    dd = int(ddof)
    if dd not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, got {dd}.")
    v1 = float(np.var(reps, ddof=1))
    v0 = float(np.var(reps, ddof=0))
    v = v1 if dd == 1 else v0
    return RichResult(payload={
        "value": v, "se": float(np.sqrt(v)),
        "variance_ddof1": v1, "variance_ddof0": v0,
        "mean_replicate": float(np.mean(reps)),
        "bias": float(np.mean(reps) - float(T(d))),
        "replicates": reps, "B": int(Bn), "n": int(n), "ddof": dd,
        "denominator_note": "ESL (7.53) divides by B - 1, not B; the "
                            "replicates are centred at their own mean",
        "what_it_estimates": "the variance of S under sampling from the "
                             "EMPIRICAL distribution; its bearing on the "
                             "real one rests on the bootstrap being "
                             "consistent for this statistic",
        "method": "Bootstrap variance estimator, ESL (7.53)"})


def cheatsheet():
    return "wsmboo: (7.53) divides by B - 1, not B -- the replicates are centred at their own mean"
