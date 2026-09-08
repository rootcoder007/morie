# morie.fn -- function file (rootcoder007/morie)
"""Aalen-Johansen estimator for competing risks."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["aalen_johansen"]


def aalen_johansen(time, cause, n_causes=None, alpha=0.05):
    r"""Cumulative incidence in the presence of competing risks.

    .. math::
       \hat F_k(t) = \sum_{t_i \le t}
         \hat S(t_{i-1})\,\frac{d_{ki}}{n_i},
       \qquad
       \hat S(t) = \prod_{t_i\le t}\left(1-\frac{d_i}{n_i}\right)

    where :math:`d_{ki}` counts events of cause :math:`k` and
    :math:`d_i` events of ANY cause.

    The factor :math:`\hat S(t_{i-1})` is the whole point. Treating the
    competing events as censored and running Kaplan-Meier on cause
    :math:`k` gives :math:`1 - \hat S_k(t)`, which OVERSTATES the
    incidence -- sometimes grossly -- because it estimates the risk in
    a hypothetical world where the competing event cannot occur. A
    patient who dies of something else is not someone who would later
    have had the event of interest; they are someone who never will.

    ``naive_km`` computes that wrong quantity deliberately, and
    ``overstatement`` reports the gap. The two coincide only when the
    competing hazard is zero.

    The cumulative incidences over all causes plus the survival sum to
    exactly one at every time, which is the arithmetic check that the
    decomposition is coherent; ``partition_residual`` reports it.

    Parameters
    ----------
    time : array-like, shape (n,)
    cause : array-like of int, shape (n,)
        0 = censored, 1..K = the cause of the observed event.
    n_causes : int, optional
    alpha : float

    Returns
    -------
    RichResult
        ``times``, ``cif`` (K by T), ``overall_survival``, ``naive_km``,
        ``overstatement``, ``partition_residual``, ``at_risk``.

    References
    ----------
    Aalen and Johansen (1978), *Scandinavian Journal of Statistics*
    5:141-150.
    Putter, Fiocco and Geskus (2007), *Statistics in Medicine*
    26:2389-2430, on why the naive Kaplan-Meier is wrong here.

    Examples
    --------
    >>> out = aalen_johansen([1, 2, 3], [1, 2, 1])
    >>> out["cif"].shape
    (2, 3)
    """
    t = np.asarray(time, dtype=float).ravel()
    c = np.asarray(cause, dtype=int).ravel()
    n = t.size
    if c.size != n:
        raise ValueError("time and cause must agree in length.")
    if n < 1:
        raise ValueError("need at least one observation.")
    if np.any(c < 0):
        raise ValueError("cause must be 0 (censored) or a positive integer.")
    K = int(c.max()) if n_causes is None else int(n_causes)
    if K < 1:
        raise ValueError("need at least one event cause.")

    order = np.argsort(t, kind="mergesort")
    t, c = t[order], c[order]
    uniq = np.unique(t[c > 0])
    T = uniq.size
    cif = np.zeros((K, T))
    surv = np.zeros(T)
    risk = np.zeros(T, dtype=int)
    naive = np.zeros((K, T))

    S = 1.0
    acc = np.zeros(K)
    Sk = np.ones(K)
    for j, u in enumerate(uniq):
        nr = int(np.sum(t >= u))
        risk[j] = nr
        d_all = int(np.sum((t == u) & (c > 0)))
        prev_S = S
        for k in range(1, K + 1):
            dk = int(np.sum((t == u) & (c == k)))
            acc[k - 1] += prev_S * dk / nr       # S(t-) weighting
            # the naive curve censors the other causes instead
            Sk[k - 1] *= (1.0 - dk / nr)
        S *= (1.0 - d_all / nr)
        cif[:, j] = acc
        surv[j] = S
        naive[:, j] = 1.0 - Sk

    partition = np.abs(cif.sum(axis=0) + surv - 1.0)
    over = naive - cif
    return RichResult(
        payload={
            "estimate": cif,
            "times": uniq,
            "cif": cif,
            "overall_survival": surv,
            "at_risk": risk,
            "naive_km": naive,
            "overstatement": over,
            "max_overstatement": float(np.max(over)) if over.size else 0.0,
            "naive_note": (
                "naive_km treats the competing events as censoring, which "
                "estimates the risk in a world where they cannot happen and "
                "overstates the incidence; the two agree only when the "
                "competing hazard is zero"
            ),
            "partition_residual": float(np.max(partition))
            if partition.size else 0.0,
            "partition_note": (
                "the cumulative incidences over all causes plus the overall "
                "survival equal one at every time; this residual is the "
                "arithmetic check that the decomposition is coherent"
            ),
            "final_cif": cif[:, -1] if T else np.zeros(K),
            "n_causes": K,
            "n_events": int(np.sum(c > 0)),
            "n_censored": int(np.sum(c == 0)),
            "n": int(n),
            "method": "Aalen-Johansen cumulative incidence",
        }
    )


def cheatsheet():
    return (
        "msmaln: Aalen-Johansen cumulative incidence with the naive "
        "Kaplan-Meier alongside to show what it overstates"
    )


# compact alias per ledger/NAMING.md
aalenjohansen = aalen_johansen
