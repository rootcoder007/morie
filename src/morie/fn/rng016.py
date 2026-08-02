# morie.fn -- function file (rootcoder007/morie)
"""Autocorrelation function of a random process by ensemble average (Eq 3.16/3.17)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_acf_continuous"]


def rangayyan_ch3_acf_continuous(x, t1, tau):
    r"""ACF of a random process at lag :math:`\tau`, estimated over an ensemble.

    .. math::

        \phi_{xx}(t_1, t_1+\tau) = E[x(t_1)\,x(t_1+\tau)]
            = \int\!\!\int x(t_1)x(t_1+\tau)\,p_{x_1,x_2}(x_1,x_2)\,dx_1\,dx_2

    The joint PDF is unknown in practice, so the expectation is approximated
    by the ensemble average over :math:`M` realisations (Eq. 3.17):

    .. math::

        \phi_{xx}(t_1, t_1+\tau) = \lim_{M\to\infty}\frac{1}{M}
            \sum_{k=1}^{M} x_k(t_1)\,x_k(t_1+\tau)

    Parameters
    ----------
    x : array-like, shape (M, N)
        Ensemble of ``M`` realisations of the process, each ``N`` samples
        long. A 1-D input is rejected: a single realisation is not an
        ensemble, and averaging along it silently computes a *time* average
        (Eq. 3.20), which is a different quantity unless the process is
        ergodic.
    t1 : int
        Sample index :math:`t_1`.
    tau : int
        Lag :math:`\tau` in samples. May be negative.

    Returns
    -------
    RichResult
        keys: ``value`` (:math:`\phi_{xx}`), ``t1``, ``tau``, ``M``, ``n``,
        ``method``.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D, or if ``t1`` or ``t1 + tau`` is out of range.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.16) and ensemble estimate Eq. (3.17), p. 96; Figure 3.2
        illustrates the two vertical lines at :math:`t_1` and :math:`t_1+\tau`
        over ten flash-visual ERP acquisitions.

    Notes
    -----
    The book's Eq. (3.20) time-averaged ACF is the *other* estimator and is
    not what this function computes. See Section 6.3 for finite-length ACF
    estimation.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"x must be a 2-D ensemble of shape (M, N); got shape {arr.shape}. "
            "Eq. (3.17) averages ACROSS realisations, not along time -- a 1-D "
            "input would give the time average of Eq. (3.20) instead."
        )
    M, N = arr.shape
    t1 = int(t1)
    tau = int(tau)
    t2 = t1 + tau
    if not (0 <= t1 < N):
        raise ValueError(f"t1={t1} out of range for N={N} samples")
    if not (0 <= t2 < N):
        raise ValueError(f"t1+tau={t2} out of range for N={N} samples")
    value = float(np.mean(arr[:, t1] * arr[:, t2]))
    return RichResult(
        payload={
            "value": value,
            "t1": t1,
            "tau": tau,
            "M": int(M),
            "n": int(N),
            "method": "ensemble-average ACF (Rangayyan Eq 3.16, estimate Eq 3.17)",
        }
    )


def cheatsheet():
    return "rng016: phi_xx(t1,t1+tau) = E[x(t1)x(t1+tau)] by ensemble average (Eq 3.16/3.17)."
