"""Alexandersson's standard normal homogeneity test for a change point."""

from math import sqrt

from . import _array_core as np
from ._richresult import hypothesis_test_result
from ._k05core import rnorm

__all__ = ["snht"]


def _tk(x, n, xbar, sigma):
    """T_k for k = 1 .. n-1, plus the running standardised partial sums."""
    tk, s, out_s = [], 0.0, []
    for k in range(1, n):
        s += (x[k - 1] - xbar) / sigma
        out_s.append(s)
        z1 = s / k
        z2 = -s / (n - k)          # the two halves sum to zero by construction
        tk.append(k * z1 * z1 + (n - k) * z2 * z2)
    return tk, out_s


def snht(x, n_mc=1999, seed=0):
    r"""Standard normal homogeneity test (Alexandersson 1986).

    Tests a single mean shift in a normal series,

    .. math:: x_i = \mu + \Delta\,\mathbf{1}\{i > k\} + \epsilon_i ,

    by maximising over every candidate split point

    .. math:: T_k = k\,\bar z_1^{\,2} + (n-k)\,\bar z_2^{\,2},
              \qquad T = \max_{1 \le k < n} T_k ,

    where :math:`\bar z_1` and :math:`\bar z_2` are the mean
    standardised deviations before and after the split.

    Because the series is standardised by its *own* mean and standard
    deviation, ``T`` is pivotal under the null -- its distribution
    depends on ``n`` alone -- so the p-value comes from simulating
    standard normal series of the same length. There is no usable
    asymptotic form: ``T`` is a maximum over n-1 dependent statistics.

    Parameters
    ----------
    x : array-like
        The series, complete observations only.
    n_mc : int
        Monte Carlo replicates for the p-value. 0 skips it and returns
        ``nan``. Draws come from morie's own Philox stream, so the R
        mirror reproduces this number exactly.
    seed : int
        Seed for that stream.

    Returns
    -------
    RichResult
        Keys ``statistic`` (T), ``pvalue``, ``change_point`` (k, the
        index *before* which the shift is placed, 1-based), ``tk``,
        ``n``, ``n_mc``.

    Notes
    -----
    The reported change point is the argmax of ``T_k``; with a genuine
    shift it lands on the last observation of the first regime.

    References
    ----------
    Alexandersson, H. (1986). A homogeneity test applied to
    precipitation data. *Journal of Climatology*, 6, 661-675.
    Definition cross-checked against the reference implementation in
    the trend R package (``snh.test``), which uses the same overall
    mean and the n-1 denominator standard deviation.
    """
    v = [float(t) for t in np.asarray(x, dtype=float).ravel().tolist()]
    n = len(v)
    if n < 3:
        raise ValueError("need at least 3 observations.")
    xbar = sum(v) / n
    ss = sum((t - xbar) ** 2 for t in v)
    sigma = sqrt(ss / (n - 1))
    if sigma <= 0:
        raise ValueError("series is constant; no change point is identifiable.")
    tk, _ = _tk(v, n, xbar, sigma)
    stat = max(tk)
    kstar = tk.index(stat) + 1

    pval = float("nan")
    n_mc = int(n_mc)
    if n_mc > 0:
        # one independent N(0,1) series per replicate, drawn from its own
        # Philox stream so replicate j never overlaps replicate j+1.
        ge = 0
        for j in range(n_mc):
            z = rnorm(n, seed=seed, stream=j + 1)
            m = sum(z) / n
            sd = sqrt(sum((t - m) ** 2 for t in z) / (n - 1))
            if sd <= 0:
                continue
            if max(_tk(z, n, m, sd)[0]) >= stat:
                ge += 1
        pval = (ge + 1) / (n_mc + 1)

    return hypothesis_test_result(
        test_name="Standard normal homogeneity test",
        statistic=float(stat),
        pvalue=float(pval),
        extra_summary=[("n", n), ("change_point", kstar)],
        extra_payload={
            "n": n,
            "change_point": kstar,
            "tk": tk,
            "n_mc": n_mc,
            "seed": int(seed),
            "method": "Alexandersson (1986) SNHT, Monte Carlo p-value",
        },
    )


def cheatsheet():
    return "snht: Alexandersson standard normal homogeneity test (single change point)"
