# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap Donsker characterisation (in probability)."""

from . import _array_core as np

from ._kosorok import bootstrap_multiplier_process, bridge_cov
from ._richresult import RichResult

__all__ = ["kosorok_ch2_bootstrap_donsker_iff"]


def kosorok_ch2_bootstrap_donsker_iff(X, t=None, n_boot=400, rng=None, F=None):
    r"""Bootstrap characterisation of the Donsker property:

    :math:`\mathcal F` is P-Donsker **iff** :math:`\hat{\mathbb G}_n`
    converges weakly in probability to :math:`\mathbb G` in
    :math:`\ell^\infty(\mathcal F)` (multiplier or nonparametric
    bootstrap).

    An *iff*, not an implication: the bootstrap working is equivalent
    to the class being Donsker, so a failing bootstrap is evidence
    against Donsker rather than merely inconclusive.

    Returns the bootstrap covariance against the theoretical bridge
    covariance at the evaluation points -- their agreement is the
    computable content of "converges weakly to G".

    Parameters
    ----------
    X : array-like
        Sample.
    t : array-like, optional
        Evaluation points; quartiles of the unit interval by default.
    n_boot : int, default 400
        Bootstrap replications.
    rng : numpy Generator, optional
    F : callable, optional
        True CDF; uniform if omitted.

    Returns
    -------
    RichResult
        keys: ``bootstrap_cov``, ``bridge_cov``, ``max_abs_gap``,
        ``t``, ``n_boot``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (bootstrap characterisations of Donsker classes).
    """
    X = np.asarray(X, dtype=float).ravel()
    if X.size < 20:
        raise ValueError(f"need at least 20 observations, got {X.size}.")
    n_boot = int(n_boot)
    if n_boot < 20:
        raise ValueError(f"n_boot must be at least 20, got {n_boot}.")
    tt = np.array([0.25, 0.5, 0.75]) if t is None else np.atleast_1d(
        np.asarray(t, dtype=float)
    )
    rng = np.random.default_rng(0) if rng is None else rng
    draws = np.array([
        bootstrap_multiplier_process(X, tt, rng=rng, F=F) for _ in range(n_boot)
    ])
    bcov = np.cov(draws.T) if tt.size > 1 else np.array([[float(np.var(draws))]])
    theo = np.array([[float(bridge_cov(a, b, F)) for b in tt] for a in tt])
    return RichResult(
        payload={"bootstrap_cov": bcov, "bridge_cov": theo,
                 "max_abs_gap": float(np.max(np.abs(bcov - theo))),
                 "t": tt, "n_boot": n_boot,
                 "method": "Bootstrap covariance vs the bridge; the iff is two-way"}
    )


def cheatsheet():
    return "ksr040: iff -- a failing bootstrap is evidence AGAINST Donsker"
