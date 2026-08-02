# morie.fn -- function file (rootcoder007/morie)
"""Leave-one-out jackknife."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boot_jackknife"]


def boot_jackknife(x, stat):
    r"""Quenouille's (1949) and Tukey's jackknife: the :math:`n`
    leave-one-out values :math:`\hat\theta_{(i)}`, the bias estimate

    .. math:: \widehat{\mathrm{bias}}_J = (n-1)
              (\bar\theta_{(\cdot)} - \hat\theta),

    and the variance estimate

    .. math:: \widehat{\mathrm{Var}}_J = \frac{n-1}{n}\sum_i
              (\hat\theta_{(i)} - \bar\theta_{(\cdot)})^2 .

    The :math:`(n-1)` factors are the whole content -- the
    leave-one-out values huddle :math:`n-1` times closer to
    :math:`\hat\theta` than independent replicates would, and the
    inflation undoes exactly that. Dropping either factor
    understates by a factor of order :math:`n`, and the tests pin
    both against closed forms (for the mean, the jackknife variance
    equals :math:`s^2/n` EXACTLY, not asymptotically).

    The known failure is non-smooth statistics: for the median the
    jackknife variance is INCONSISTENT (Efron 1979 Sec. 3 uses it as
    the motivating example for the bootstrap), because leave-one-out
    perturbations explore only n points of a step function. The
    output names the restriction.

    Parameters
    ----------
    x : array-like
        Sample; rows are observations.
    stat : callable
        The statistic; must be smooth in the observations for the
        variance to mean anything.

    Returns
    -------
    RichResult
        keys: ``leave_one_out``, ``estimate``, ``bias``,
        ``corrected``, ``variance``, ``se``, ``pseudovalues``, ``n``,
        ``smoothness_caveat``, ``method``.

    References
    ----------
    Quenouille, M. H. (1949), "Approximate tests of correlation in
    time-series", *JRSS-B* 11:68-84. Tukey, J. W. (1958), *Annals of
    Mathematical Statistics* 29:614. Efron (1979), Sec. 3.
    """
    d = np.asarray(x, dtype=float)
    n = d.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    th = float(stat(d))
    loo = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        loo[i] = float(stat(d[idx != i]))
    m = float(loo.mean())
    bias = (n - 1) * (m - th)
    var = (n - 1) / n * float(np.sum((loo - m) ** 2))
    pseudo = n * th - (n - 1) * loo
    return RichResult(payload={
        "leave_one_out": loo, "estimate": th,
        "bias": bias, "corrected": th - bias,
        "variance": var, "se": float(np.sqrt(var)),
        "pseudovalues": pseudo,
        "inflation_note": "both (n-1) factors undo the leave-one-out "
                          "values' huddling; dropping either understates "
                          "by a factor of order n",
        "smoothness_caveat": "inconsistent for non-smooth statistics -- the "
                             "median is the canonical failure (Efron 1979 "
                             "Sec. 3); use the bootstrap there",
        "n": int(n),
        "method": "Leave-one-out jackknife (Quenouille 1949; Tukey 1958)"})


def cheatsheet():
    return "btjkn: the (n-1) factors ARE the estimator -- and the median breaks it"
