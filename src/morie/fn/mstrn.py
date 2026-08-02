# morie.fn -- function file (rootcoder007/morie)
"""Aalen-Johansen multistate transition probabilities."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["multistate_transition_matrix"]


def multistate_transition_matrix(time, state_from, state_to, n_states=None, s=0.0, t=None):
    r"""Aalen-Johansen estimator of multistate transition probabilities.

    .. math:: \hat P(s, t) = \prod_{s < u \le t}\big(I + d\hat A(u)\big),

    a product integral over event times, where
    :math:`d\hat A_{jk}(u) = dN_{jk}(u)/Y_j(u)` is the observed number
    of j-to-k transitions divided by the number at risk in j, and the
    diagonal is set so each row of :math:`d\hat A` sums to zero.

    This is nonparametric: it assumes no Markov transition-intensity
    model, only that censoring is independent. Rows of the result sum
    to one by construction, which the tests check.

    Parameters
    ----------
    time : array-like, shape (k,)
        Transition times.
    state_from, state_to : array-like of int, shape (k,)
        Origin and destination states, 0-indexed.
    n_states : int, optional
        Total states; inferred from the data if omitted.
    s : float, default 0.0
        Start of the interval.
    t : float, optional
        End of the interval; defaults to the last event time.

    Returns
    -------
    RichResult
        keys: ``P`` (n_states, n_states), ``event_times``,
        ``increments`` (list of dA matrices), ``at_risk``,
        ``n_states``, ``n_transitions``, ``s``, ``t``, ``method``.

    References
    ----------
    Aalen, O. O. & Johansen, S. (1978). An empirical transition matrix
    for non-homogeneous Markov chains based on censored observations.
    *Scandinavian Journal of Statistics*, 5(3), 141-150.

    Andersen, P. K., Borgan, O., Gill, R. D. & Keiding, N. (1993).
    *Statistical Models Based on Counting Processes*. Springer. Ch. IV.
    """
    time = np.asarray(time, dtype=float).ravel()
    sf = np.asarray(state_from, dtype=int).ravel()
    st = np.asarray(state_to, dtype=int).ravel()
    k = time.size
    if not (sf.size == k and st.size == k):
        raise ValueError("time, state_from and state_to must have the same length.")
    if k < 2:
        raise ValueError(f"need at least 2 transitions, got {k}.")
    if np.any(sf < 0) or np.any(st < 0):
        raise ValueError("states must be non-negative integers.")
    ns = int(n_states) if n_states is not None else int(max(sf.max(), st.max()) + 1)
    if ns < 2:
        raise ValueError("need at least 2 states.")
    if np.any(sf >= ns) or np.any(st >= ns):
        raise ValueError(f"state labels must be below n_states = {ns}.")
    t = float(time.max()) if t is None else float(t)
    if t <= s:
        raise ValueError(f"t must exceed s, got s = {s}, t = {t}.")

    order = np.argsort(time)
    time, sf, st = time[order], sf[order], st[order]
    # occupancy: everyone starts in their first observed origin state
    at_risk = np.bincount(sf[np.unique(time, return_index=True)[1]], minlength=ns).astype(
        float
    )
    at_risk = np.bincount(sf, minlength=ns).astype(float)

    P = np.eye(ns)
    times = []
    incs = []
    occ = at_risk.copy()
    for u in np.unique(time):
        if not (s < u <= t):
            continue
        mask = time == u
        dA = np.zeros((ns, ns))
        for j, kk in zip(sf[mask], st[mask]):
            if occ[j] > 0:
                dA[j, kk] += 1.0 / occ[j]
        np.fill_diagonal(dA, 0.0)
        dA[np.diag_indices(ns)] = -dA.sum(axis=1)
        P = P @ (np.eye(ns) + dA)
        for j, kk in zip(sf[mask], st[mask]):
            occ[j] -= 1
            occ[kk] += 1
        times.append(float(u))
        incs.append(dA)

    return RichResult(
        payload={
            "P": P, "event_times": np.array(times), "increments": incs,
            "at_risk": at_risk, "n_states": ns, "n_transitions": int(k),
            "s": float(s), "t": t,
            "method": "Aalen-Johansen product-integral transition matrix",
        }
    )


def cheatsheet():
    return "mstrn: P(s,t) = prod (I + dA(u)); nonparametric, rows sum to 1"
