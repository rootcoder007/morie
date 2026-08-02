# morie.fn -- function file (rootcoder007/morie)
"""MSE cost function of the Wiener filter (Rangayyan Eq 3.166)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_mse_cost_function"]


def rangayyan_ch3_mse_cost_function(w, Theta, Phi, sigma_d):
    r"""Mean-squared-error cost of a tap-weight vector under Wiener filter theory.

    .. math::

        J(\mathbf{w}) = E[e^2(n)]
            = \sigma_d^2 - \mathbf{w}^T\Theta - \Theta^T\mathbf{w}
              + \mathbf{w}^T\Phi\mathbf{w}

    Parameters
    ----------
    w : array-like, shape (M,)
        Tap-weight vector.
    Theta : array-like, shape (M,)
        Cross-correlation vector (Eq. 3.160); see :mod:`morie.fn.rng142`.
    Phi : array-like, shape (M, M)
        Autocorrelation matrix (Eq. 3.163); see :mod:`morie.fn.rng143`.
    sigma_d : float
        Standard deviation :math:`\sigma_d` of the desired response, whose
        mean is assumed zero. **This is the SD, not the variance** -- the
        book writes :math:`E[d^2(n)]` as :math:`\sigma_d^2`.

    Returns
    -------
    RichResult
        keys: ``value`` (:math:`J(\mathbf{w})`), ``M``, ``method``.

    Raises
    ------
    ValueError
        On any shape mismatch, or if ``sigma_d`` is negative.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.159) sets up :math:`J(\mathbf{w}) = E[e^2(n)]` on p. 174; the
        simplified quadratic form used here is **Eq. (3.166), p. 175**. The
        module previously cited "Eq 3.159 / 3.166, p. 174" -- 3.166 is on
        p. 175.

    Notes
    -----
    :math:`J` is a second-order function of :math:`\mathbf{w}` with minimum
    :math:`J_{\min} = \sigma_d^2 - \Theta^T\Phi^{-1}\Theta` (Eq. 3.172) at
    :math:`\mathbf{w}_o = \Phi^{-1}\Theta` (Eq. 3.169). That pair is what the
    tests pin, since it checks the quadratic and the Wiener-Hopf solution
    against each other rather than against a transcribed constant.
    """
    wv = np.asarray(w, dtype=float).ravel()
    th = np.asarray(Theta, dtype=float).ravel()
    ph = np.asarray(Phi, dtype=float)
    sd = float(sigma_d)
    M = wv.size
    if th.size != M:
        raise ValueError(f"w and Theta must have the same length; got {M} and {th.size}")
    if ph.shape != (M, M):
        raise ValueError(f"Phi must have shape ({M}, {M}); got {ph.shape}")
    if not np.isfinite(sd) or sd < 0:
        raise ValueError(
            f"sigma_d must be finite and non-negative; got {sigma_d!r}. It is the "
            "SD of the desired response, not its variance."
        )
    J = sd**2 - wv @ th - th @ wv + wv @ ph @ wv
    return RichResult(
        payload={
            "value": float(J),
            "M": int(M),
            "method": "Wiener MSE cost J(w) (Rangayyan Eq 3.166)",
        }
    )


def cheatsheet():
    return "rng141: J(w) = sigma_d^2 - w'Theta - Theta'w + w'Phi w (Rangayyan Eq 3.166)."
