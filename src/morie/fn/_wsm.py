# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the plug-in / resampling / Monte Carlo shelf.

Every module in this group is a standard piece of nonparametric
inference. The placeholder docstrings cited a textbook that is not in
this repository's reference library, so each has been re-grounded on
a primary source that IS there and that was read from the PDF:

* Silverman, B. W. (1986), *Density Estimation for Statistics and
  Data Analysis*, Chapman and Hall -- the kernel estimator (2.2a)
  and the bandwidth rules (3.28)-(3.31).
* MacKay, D. J. C. (2003), *Information Theory, Inference, and
  Learning Algorithms*, Cambridge University Press -- importance
  sampling, Sec. 29.2, Eqs. (29.21)-(29.22).
* Hastie, Tibshirani and Friedman (2009), *The Elements of
  Statistical Learning*, 2nd ed. -- the bootstrap variance (7.53)
  and bagging (Sec. 8.7).
* Kosorok, M. R. (2008), *Introduction to Empirical Processes and
  Semiparametric Inference*, Springer -- the functional delta method
  (Ch. 12), the bootstrap for Donsker classes (Ch. 10) and
  M-estimation (Ch. 14).

Where no source in the library covers a topic, the module says so
rather than inventing a citation.
"""

from . import _array_core as np

__all__ = ["silverman_bandwidth", "adaptive_spread", "bootstrap_replicates"]


def adaptive_spread(x):
    r"""Silverman's Eq. (3.30):
    :math:`A = \min(\text{standard deviation},
    \text{interquartile range}/1.34)`.

    The divisor is 1.34 as the book prints it (the normal-theory
    value is 1.349; Silverman rounds). Using the plain standard
    deviation instead is what makes the rule oversmooth long-tailed
    and skewed data -- a single outlier moves the standard deviation
    and barely touches the interquartile range.
    """
    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 2:
        raise ValueError(f"need at least 2 observations, got {xv.size}.")
    sd = float(np.std(xv, ddof=1))
    iqr = float(np.subtract(*np.percentile(xv, [75, 25])))
    if iqr > 0:
        return min(sd, iqr / 1.34)
    return sd


def silverman_bandwidth(x, rule="3.31"):
    r"""The window-width rules of Silverman Sec. 3.4.2.

    ``"3.28"``  :math:`h = 1.06\,\sigma\,n^{-1/5}` -- the pure normal
    reference. Optimal for normal data and, in the book's words,
    it "oversmooths even further" for bimodal densities.

    ``"3.29"``  :math:`h_{opt} = 0.79\,R\,n^{-1/5}` with :math:`R`
    the interquartile range -- better for long-tailed and skewed
    data, worse for bimodal.

    ``"3.31"``  :math:`h = 0.9\,A\,n^{-1/5}` with :math:`A` from
    (3.30) -- **the book's actual recommendation**, and the default
    here. Silverman reports it lands within 10% of the optimal mean
    integrated square error for every t-distribution considered, for
    log-normals with skewness up to about 1.8, and for normal
    mixtures separated by up to 3 standard deviations.

    The constant is 0.9 and not 1.06. A great deal of code uses
    1.06 -- that is (3.28), which the book presents as the starting
    point it then improves on twice, not as the recommendation.
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if rule == "3.28":
        s = float(np.std(xv, ddof=1))
        return 1.06 * (s if s > 0 else 1.0) * n ** (-0.2)
    if rule == "3.29":
        r = float(np.subtract(*np.percentile(xv, [75, 25])))
        return 0.79 * (r if r > 0 else 1.0) * n ** (-0.2)
    if rule == "3.31":
        a = adaptive_spread(xv)
        return 0.9 * (a if a > 0 else 1.0) * n ** (-0.2)
    raise ValueError("rule must be '3.28', '3.29' or '3.31'.")


def bootstrap_replicates(data, statistic, B=1000, seed=0):
    """``B`` values of ``statistic`` on samples drawn with
    replacement (Efron 1979; ESL Fig. 7.12)."""
    d = np.asarray(data, dtype=float)
    n = d.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    B = int(B)
    if B < 2:
        raise ValueError(f"need at least 2 replicates, got {B}.")
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = float(statistic(d[rng.integers(0, n, n)]))
    return out


def cheatsheet():
    return ("_wsm: Silverman's rule is 0.9 A n^(-1/5) (3.31), NOT 1.06 sigma "
            "n^(-1/5) (3.28) -- the second is what he improves on")
