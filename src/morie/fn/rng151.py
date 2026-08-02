# morie.fn -- function file (rootcoder007/morie)
"""Optimal Wiener filter for noise removal (Rangayyan Eq 3.183)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_optimal_for_noise_removal"]


def rangayyan_ch3_wiener_optimal_for_noise_removal(Phi_d, Phi_eta, Phi_1d):
    r"""Optimal Wiener tap-weight vector when the input is signal plus noise.

    For :math:`x(n) = d(n) + \eta(n)` with signal and noise statistically
    independent and at least one of zero mean, Eq. (3.181) gives
    :math:`\Phi = \Phi_d + \Phi_\eta` and Eq. (3.182) gives
    :math:`\Theta = \Phi_{1d}`, so Eq. (3.169) becomes

    .. math::

        \mathbf{w}_o = (\Phi_d + \Phi_\eta)^{-1}\,\Phi_{1d}.

    Parameters
    ----------
    Phi_d : array-like, shape (M, M)
        Autocorrelation matrix of the desired signal.
    Phi_eta : array-like, shape (M, M)
        Autocorrelation matrix of the noise.
    Phi_1d : array-like, shape (M,)
        :math:`M \times 1` autocorrelation vector of the desired signal.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\mathbf{w}_o`), ``M``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch, or if :math:`\Phi_d + \Phi_\eta` is singular.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.183), p. 177, via Eq. (3.181) and Eq. (3.182) on the same page.

    Notes
    -----
    Solved with :func:`numpy.linalg.solve` rather than by forming the inverse:
    the sum of two autocorrelation matrices is positive definite in theory but
    can be badly conditioned in practice, and an explicit inverse loses more
    digits than a solve. A singular sum raises rather than returning ``inf``,
    because Eq. (3.183) has no solution in that case.
    """
    Pd = np.asarray(Phi_d, dtype=float)
    Pe = np.asarray(Phi_eta, dtype=float)
    P1 = np.asarray(Phi_1d, dtype=float).ravel()
    if Pd.ndim != 2 or Pd.shape[0] != Pd.shape[1]:
        raise ValueError(f"Phi_d must be a square matrix; got shape {Pd.shape}")
    if Pe.shape != Pd.shape:
        raise ValueError(f"Phi_eta must match Phi_d shape {Pd.shape}; got {Pe.shape}")
    M = Pd.shape[0]
    if P1.size != M:
        raise ValueError(f"Phi_1d must have length {M}; got {P1.size}")
    try:
        w_o = np.linalg.solve(Pd + Pe, P1)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Phi_d + Phi_eta is singular, so Eq. (3.183) has no solution. "
            "Check that both are genuine autocorrelation matrices."
        ) from exc
    return RichResult(
        payload={
            "array": w_o,
            "M": int(M),
            "method": "optimal Wiener filter for noise removal (Rangayyan Eq 3.183)",
        }
    )


def cheatsheet():
    return "rng151: w_o = (Phi_d + Phi_eta)^-1 Phi_1d (Rangayyan Eq 3.183)."
