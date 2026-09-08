# morie.fn -- function file (rootcoder007/morie)
"""Individual treatment effect (ITE) using potential outcomes notation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["potential_outcomes_individual"]


def potential_outcomes_individual(Y1, Y0, observed_treatment=None):
    r"""Individual effects :math:`\tau_i = Y_i(1) - Y_i(0)` and what is observable.

    Given both potential-outcome vectors (a simulation or a thought
    experiment -- in real data at most one is ever observed per unit,
    Holland's fundamental problem of causal inference), returns the
    ITE vector, the ATE, and, when an observed-treatment vector is
    supplied, the naive observed-difference estimate and its bias
    relative to the true ATE.

    Parameters
    ----------
    Y1, Y0 : array-like, shape (n,)
        Potential outcomes under treatment and control.
    observed_treatment : array-like of {0, 1}, optional
        Which potential outcome each unit reveals.

    Returns
    -------
    RichResult
        keys: ``ite`` (n,), ``ate``, ``ite_var``, ``naive_diff``
        (None without ``observed_treatment``), ``selection_bias``,
        ``n``, ``method``.

    References
    ----------
    Holland, P. W. (1986). Statistics and causal inference. *Journal
    of the American Statistical Association*, 81(396), 945-960. (the
    fundamental problem; the decomposition of the naive contrast)
    """
    Y1 = np.asarray(Y1, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float).ravel()
    if Y1.size != Y0.size:
        raise ValueError("Y1 and Y0 must have equal length.")
    ite = Y1 - Y0
    ate = float(ite.mean())

    naive = bias = None
    if observed_treatment is not None:
        T = np.asarray(observed_treatment, dtype=float).ravel()
        if T.size != Y1.size or not np.all(np.isin(T, (0.0, 1.0))):
            raise ValueError("observed_treatment must be binary 0/1 of matching length.")
        if T.min() == T.max():
            raise ValueError("observed_treatment needs both arms.")
        naive = float(Y1[T == 1].mean() - Y0[T == 0].mean())
        bias = naive - ate

    return RichResult(
        payload={
            "ite": ite,
            "ate": ate,
            "ite_var": float(ite.var(ddof=1)) if ite.size > 1 else float("nan"),
            "naive_diff": naive,
            "selection_bias": bias,
            "n": int(Y1.size),
            "method": "Individual treatment effects tau_i = Y_i(1) - Y_i(0)",
        }
    )


def cheatsheet():
    return "potef: ITE vector, ATE, and the naive-contrast selection bias (Holland 1986)"
