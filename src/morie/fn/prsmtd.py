# morie.fn -- function file (rootcoder007/morie)
"""Sequential propensity-score matching."""

from . import _array_core as np

from ._richresult import RichResult
from .aiptdd import _logit_fit

__all__ = ["propensity_score_method"]


def propensity_score_method(A, H, time=None):
    r"""Sequential (risk-set) propensity matching for time-varying treatment.

    At each period t, a time-t propensity is fitted among the units
    still untreated at t-1 -- the risk set -- by logistic regression of
    initiation :math:`A_t` on the current history :math:`H_t`. Each
    initiator is greedily matched to the not-yet-treated unit in the
    same risk set with the nearest propensity, without replacement
    within the period. This is the balancing-score idea behind Lu's
    time-dependent propensity matching (there via a Cox model; here via
    period-by-period logistic risk sets).

    Parameters
    ----------
    A : array-like of {0, 1}, shape (n, T) or (n,)
        Treatment/initiation indicator per period. Treatment is
        absorbing: once 1, later values are ignored.
    H : array-like, shape (n, T) or (n,)
        Time-varying covariate history (value entering period t).
    time : ignored
        Accepted for backward compatibility with the placeholder
        signature.

    Returns
    -------
    RichResult
        keys: ``matched_idx`` (m, 3) array of (period, initiator,
        matched control) rows, ``n_matched``, ``n_initiators``,
        ``propensity`` (n, T), ``n``, ``method``.

    References
    ----------
    Lu, B. (2005). Propensity score matching with time-dependent
    covariates. *Biometrics*, 61(3), 721-728.
    doi:10.1111/j.1541-0420.2005.00356.x.
    """
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if H.ndim == 1:
        H = H[:, None]
    n, T = A.shape
    if H.shape != (n, T):
        raise ValueError(f"A and H must share shape, got {A.shape} and {H.shape}.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")

    ever = np.zeros(n, dtype=bool)
    used_control = np.zeros(n, dtype=bool)  # matching without replacement
    ps = np.full((n, T), np.nan)
    pairs = []
    n_init = 0
    for t in range(T):
        risk = ~ever
        a_t = (A[:, t] == 1) & risk
        n_init += int(a_t.sum())
        if risk.sum() >= 4 and a_t.sum() > 0 and a_t.sum() < risk.sum():
            idx = np.flatnonzero(risk)
            e = np.clip(_logit_fit(H[idx, t][:, None], A[idx, t]), 1e-6, 1 - 1e-6)
            ps[idx, t] = e
            initiators = idx[A[idx, t] == 1]
            controls = list(idx[(A[idx, t] == 0) & ~used_control[idx]])
            for i in initiators[np.argsort(-ps[initiators, t])]:
                if not controls:
                    break
                d = np.abs(ps[controls, t] - ps[i, t])
                j = controls.pop(int(np.argmin(d)))
                used_control[j] = True
                pairs.append((t, int(i), int(j)))
        ever |= A[:, t] == 1

    matched = np.array(pairs, dtype=int).reshape(-1, 3)
    return RichResult(
        payload={
            "matched_idx": matched,
            "n_matched": int(matched.shape[0]),
            "n_initiators": int(n_init),
            "propensity": ps,
            "n": int(n),
            "method": "Sequential propensity-score matching (period risk sets)",
        }
    )


def cheatsheet():
    return "prsmtd: risk-set propensity matching per period (Lu 2005 idea, logistic)"
