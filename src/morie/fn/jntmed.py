# morie.fn -- function file (rootcoder007/morie)
"""Joint significance test for mediation."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["joint_significance_mediation"]


def joint_significance_mediation(x, m, y, alpha=0.05):
    r"""Joint significance ("max-p") test of the mediated path.

    Fits M = i1 + a X and Y = i2 + c'X + b M by OLS and declares
    mediation when BOTH a and b are individually significant; the
    p-value reported is max(p_a, p_b). In the MacKinnon et al. (2002)
    comparison of 14 mediation tests this simple procedure held the
    best balance of Type I error and power -- better than the Sobel
    test, whose normal approximation to the product ab is poor exactly
    where mediation is in doubt.

    This replaces a placeholder that averaged its first argument.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    alpha : float, default 0.05
        Per-path significance level.

    Returns
    -------
    RichResult
        keys: ``significant``, ``p_value`` (= max(p_a, p_b)), ``a``,
        ``b``, ``p_a``, ``p_b``, ``indirect`` (= ab), ``n``,
        ``method``.

    References
    ----------
    MacKinnon, D. P., Lockwood, C. M., Hoffman, J. M., West, S. G. &
    Sheets, V. (2002). A comparison of methods to test mediation and
    other intervening variable effects. *Psychological Methods*, 7(1),
    83-104.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not (m.size == n and y.size == n):
        raise ValueError(f"x, m and y must share a length; got {n}, {m.size}, {y.size}.")
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    def _ols_t(D, resp, col):
        beta, *_ = np.linalg.lstsq(D, resp, rcond=None)
        r = resp - D @ beta
        dof = n - D.shape[1]
        s2 = float(r @ r) / dof
        cov = s2 * np.linalg.inv(D.T @ D)
        t = beta[col] / np.sqrt(cov[col, col])
        return float(beta[col]), float(2 * stats.t.sf(abs(t), dof))

    a, p_a = _ols_t(np.column_stack([np.ones(n), x]), m, 1)
    b, p_b = _ols_t(np.column_stack([np.ones(n), x, m]), y, 2)
    p = max(p_a, p_b)
    return RichResult(
        payload={
            "significant": bool(p < alpha),
            "p_value": p,
            "a": a,
            "b": b,
            "p_a": p_a,
            "p_b": p_b,
            "indirect": a * b,
            "n": int(n),
            "alpha": float(alpha),
            "method": "Joint significance (max-p) mediation test (MacKinnon et al. 2002)",
        }
    )


def cheatsheet():
    return "jntmed: joint-significance (max-p) mediation test (MacKinnon et al. 2002)"
