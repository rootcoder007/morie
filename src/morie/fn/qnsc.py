# morie.fn -- function file (rootcoder007/morie)
"""Qn robust scale estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["qn_scale"]


def qn_scale(x):
    r"""The Qn scale estimator of Rousseeuw and Croux (1993),

    .. math:: Q_n = d\,\{|x_i - x_j| : i < j\}_{(k)},
              \qquad k = \binom{h}{2},\ h = \lfloor n/2\rfloor + 1,

    the k-th order statistic of the :math:`\binom n2` pairwise
    absolute differences, with
    :math:`d = 1/(\sqrt2\,\Phi^{-1}(5/8)) \approx 2.2191` for
    consistency at the normal.

    Qn is the answer to a specific complaint about the MAD: the MAD
    is built around a location (the median) and is therefore aimed at
    SYMMETRIC distributions, and its normal efficiency is only 37%.
    Qn uses no location at all -- differences only -- has the same
    50% breakdown point, and reaches 82% efficiency. The paper's
    small-sample correction factors are applied for n < 10 and the
    asymptotic 1 + 1.4/n (n odd) or 1 + 3.7/n (n even) style factors
    beyond, as tabulated in Sec. 4; without them Qn is biased low in
    exactly the small samples robust scales get used on.

    Parameters
    ----------
    x : array-like
        Sample, at least 2 observations.

    Returns
    -------
    RichResult
        keys: ``value``, ``k``, ``h``, ``d``, ``correction``,
        ``breakdown``, ``gaussian_efficiency``, ``location_free``,
        ``n``, ``method``.

    References
    ----------
    Rousseeuw, P. J. and Croux, C. (1993), "Alternatives to the
    median absolute deviation", *JASA* 88:1273-1283, Secs. 2-4.
    """
    from ._robust import QN_D

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    h = n // 2 + 1
    k = h * (h - 1) // 2
    diffs = np.abs(xv[:, None] - xv[None, :])[np.triu_indices(n, 1)]
    stat = float(np.partition(diffs, k - 1)[k - 1])
    # finite-sample corrections, Rousseeuw and Croux Sec. 4
    small = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844,
             6: 0.611, 7: 0.857, 8: 0.669, 9: 0.872}
    if n <= 9:
        corr = small[n]
    elif n % 2 == 1:
        corr = n / (n + 1.4)
    else:
        corr = n / (n + 3.8)
    value = QN_D * corr * stat
    return RichResult(payload={
        "value": value, "k": int(k), "h": int(h), "d": QN_D,
        "correction": float(corr),
        "breakdown": 0.5, "gaussian_efficiency": 0.82,
        "location_free": True,
        "versus_mad": "no location is used at all, so Qn is not aimed at "
                      "symmetric distributions the way the MAD is, and its "
                      "normal efficiency is 82% against the MAD's 37%",
        "n": int(n),
        "method": "Qn = d * k-th order statistic of pairwise |differences| "
                  "(Rousseeuw-Croux 1993)"})


def cheatsheet():
    return "qnsc: pairwise differences, no location -- 50% breakdown at 82% efficiency"
