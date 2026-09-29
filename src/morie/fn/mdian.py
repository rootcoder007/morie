# morie.fn -- function file (rootcoder007/morie)
"""Mediation analysis front-end (total = direct + indirect)."""

__all__ = ["mediation_analysis"]


def mediation_analysis(Y, T, M, X=None):
    r"""Decompose a total effect into direct and indirect parts.

    Delegates to :func:`morie.fn.bkmed.baron_kenny` on (Y, T, M); the
    optional baseline covariates X are residualised out of all three
    variables first (Frisch-Waugh), which leaves the paths unchanged in
    the linear model.

    References
    ----------
    Baron, R. M. & Kenny, D. A. (1986). The moderator-mediator
    variable distinction in social psychological research: conceptual,
    strategic, and statistical considerations. *J. Pers. Soc.
    Psychol.*, 51(6), 1173-1182.

    Examples
    --------
    >>> import math
    >>> X = [math.sin(k) for k in range(12)]
    >>> M = [0.5 * X[k] + 0.3 * math.cos(2 * k) for k in range(12)]
    >>> Y = [0.4 * X[k] + 0.8 * M[k] + 0.2 * math.sin(3 * k) for k in range(12)]
    >>> round(mediation_analysis(Y, X, M)["indirect"], 12)
    0.383604604142
    """
    from morie.fn import _array_core as np

    from .bkmed import baron_kenny

    y = np.asarray(Y, dtype=float).ravel()
    t = np.asarray(T, dtype=float).ravel()
    m = np.asarray(M, dtype=float).ravel()
    if X is not None:
        C = np.asarray(X, dtype=float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        D = np.column_stack([np.ones(len(y)), C])

        def resid(v):
            return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]

        y, t, m = resid(y), resid(t), resid(m)
    return baron_kenny(y, t, m)


def cheatsheet():
    return "mdian: mediation decomposition (front-end to bkmed, covariates residualised)"
