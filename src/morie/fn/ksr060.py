# morie.fn -- function file (rootcoder007/morie)
"""U-process."""

import numpy as np

from itertools import combinations

from ._richresult import RichResult

__all__ = ["kosorok_ch2_u_process_measure"]


def kosorok_ch2_u_process_measure(f, X, m=2, n=None):
    r"""U-statistic / U-process of order m (Kosorok Ch. 2):

    .. math:: U_{n,m}(f) = \binom{n}{m}^{-1}
              \sum_{i_1 < \cdots < i_m} f(X_{i_1}, \dots, X_{i_m}).

    Averages a symmetric kernel over all m-subsets, so unlike an
    empirical mean the summands are DEPENDENT -- which is why
    U-processes need their own maximal inequalities rather than
    inheriting the empirical-process ones.

    Also returns the Hajek projection variance
    :math:`m^2 \zeta_1 / n`, the first-order term of the U-statistic
    variance, so the dependence is quantified rather than merely
    noted.

    Parameters
    ----------
    f : callable
        Kernel taking m arguments.
    X : array-like
        Sample.
    m : int, default 2
        Kernel order.
    n : int, optional
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``U``, ``n_subsets``, ``zeta1``, ``hajek_var``, ``m``,
        ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (U-processes).
    """
    from math import comb

    X = np.asarray(X, dtype=float).ravel()
    N = X.size
    m = int(m)
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}.")
    if N < m:
        raise ValueError(f"need at least m = {m} observations, got {N}.")
    if comb(N, m) > 200000:
        raise ValueError(
            f"C({N}, {m}) = {comb(N, m)} subsets is too many to enumerate "
            "exactly; subsample first."
        )
    vals = np.array([f(*X[list(idx)]) for idx in combinations(range(N), m)],
                    dtype=float)
    U = float(vals.mean())
    # zeta1 = Var(E[f(X1,...,Xm) | X1]), estimated by conditioning on
    # each observation in turn
    g = np.zeros(N)
    cnt = np.zeros(N)
    for v, idx in zip(vals, combinations(range(N), m)):
        for i in idx:
            g[i] += v
            cnt[i] += 1
    g = np.where(cnt > 0, g / np.maximum(cnt, 1), U)
    zeta1 = float(np.var(g, ddof=1)) if N > 1 else 0.0
    return RichResult(
        payload={"U": U, "n_subsets": int(comb(N, m)), "zeta1": zeta1,
                 "hajek_var": float(m**2 * zeta1 / N), "m": m, "n": int(N),
                 "method": "U_{n,m}(f) over all m-subsets, with the Hajek variance"}
    )


def cheatsheet():
    return "ksr060: U-statistic over C(n,m) subsets; summands dependent, Hajek var given"
