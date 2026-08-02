# morie.fn -- function file (rootcoder007/morie)
"""RLS correlation matrix."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_phi_matrix"]


def rangayyan_ch3_rls_phi_matrix(r, lam=0.99, n=None):
    r"""RLS exponentially weighted correlation matrix (Rangayyan
    Ch. 3):

    .. math:: \Phi(n) = \sum_{i=1}^{n} \lambda^{n-i}\,
              \mathbf{r}(i)\,\mathbf{r}^T(i).

    The forgetting factor lambda < 1 discounts old data geometrically,
    giving an effective memory of about 1/(1 - lambda) samples --
    returned, because that number, not n, is what governs how fast RLS
    tracks a change.

    Parameters
    ----------
    r : array-like, shape (N, p)
        Reference vectors.
    lam : float in (0, 1], default 0.99
        Forgetting factor.
    n : int, optional
        Time index; the last sample if omitted.

    Returns
    -------
    RichResult
        keys: ``Phi`` (p, p), ``effective_memory``, ``lam``, ``n``,
        ``condition_number``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the RLS algorithm).
    """
    R = np.atleast_2d(np.asarray(r, dtype=float))
    lam = float(lam)
    if not 0 < lam <= 1:
        raise ValueError(f"lam must lie in (0, 1], got {lam}.")
    N = R.shape[0]
    idx = N if n is None else int(n)
    if not 1 <= idx <= N:
        raise ValueError(f"n must lie in 1..{N}, got {idx}.")
    w = lam ** (idx - 1 - np.arange(idx))
    Phi = (R[:idx] * w[:, None]).T @ R[:idx]
    mem = np.inf if lam == 1.0 else 1.0 / (1.0 - lam)
    try:
        cond = float(np.linalg.cond(Phi))
    except np.linalg.LinAlgError:
        cond = np.inf
    return RichResult(payload={"Phi": Phi, "effective_memory": float(mem),
                               "lam": lam, "n": idx, "condition_number": cond,
                               "method": "Phi(n) = sum lambda^(n-i) r(i) r^T(i)"})


def cheatsheet():
    return "rng165: memory ~ 1/(1-lambda) governs tracking, not n"
