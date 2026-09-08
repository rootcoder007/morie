# morie.fn -- function file (rootcoder007/morie)
"""MCMC autocorrelation diagnostic and effective sample size (Geyer 1992)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["autocorrelation_check"]


def autocorrelation_check(chain, max_lag=None):
    r"""Autocorrelation function and effective sample size of an MCMC chain.

    .. math::

        \rho_k = \frac{\sum_{t=1}^{N-k}(x_t-\bar{x})(x_{t+k}-\bar{x})}
                      {\sum_{t=1}^{N}(x_t-\bar{x})^2},
        \qquad
        N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^{K}\rho_k}

    where :math:`K` is chosen by Geyer's **initial positive sequence**: the
    lags are paired as :math:`\Gamma_m = \rho_{2m} + \rho_{2m+1}`, and the sum
    is truncated at the first :math:`m` for which :math:`\Gamma_m \le 0`.

    Parameters
    ----------
    chain : array-like, shape (N,)
        Draws from a single chain, in sampling order.
    max_lag : int, optional
        Hard cap on the lag search. Defaults to ``N - 1``.

    Returns
    -------
    RichResult
        keys: ``estimate`` (:math:`N_{\text{eff}}`), ``ess``, ``acf`` (lags
        ``0..K``), ``sum_rho``, ``truncation_lag`` (:math:`K`),
        ``efficiency`` (:math:`N_{\text{eff}}/N`), ``n``, ``method``.

    Raises
    ------
    ValueError
        If ``chain`` is not 1-D, has fewer than 4 draws, or is constant --
        a chain that never moved has no autocorrelation structure and an
        undefined ESS, which is a *failure* to report, not zero.

    References
    ----------
    Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
        *Statistical Science*, 7(4), 473-483. The initial positive sequence
        estimator, Section 3.3.

    Notes
    -----
    Truncation is the whole difficulty. The empirical :math:`\rho_k` are
    noisy at large :math:`k` and each contributes variance but no signal, so
    summing all :math:`N-1` of them gives an estimator that does not converge.
    Geyer's insight is that for a reversible chain the *paired* sums
    :math:`\Gamma_m` are positive, so the first non-positive pair marks where
    the estimates have become noise.

    :math:`N_{\text{eff}}` can exceed :math:`N` when the chain is
    antithetic (negative autocorrelation), which is a real property of some
    samplers and is not clipped here.

    A high ESS is necessary but not sufficient for convergence: a chain stuck
    in one mode can be almost independent within that mode and still be
    entirely wrong about the target.
    """
    x = np.asarray(chain, dtype=float).ravel()
    n = x.size
    if np.asarray(chain).ndim > 1 and np.asarray(chain).shape[0] != n:
        raise ValueError(
            f"chain must be 1-D (a single chain in sampling order); got shape "
            f"{np.asarray(chain).shape}"
        )
    if n < 4:
        raise ValueError(f"need at least 4 draws to estimate autocorrelation; got {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError("chain must be finite")
    dev = x - x.mean()
    denom = float(dev @ dev)
    if denom == 0.0:
        raise ValueError(
            "chain is constant -- it never moved. The ESS is undefined, which is "
            "a sampler failure to report rather than a number to return."
        )
    cap = (n - 1) if max_lag is None else min(int(max_lag), n - 1)
    # Autocovariance for every lag in one FFT, O(n log n), instead of a Python
    # loop of `cap` dot products which is O(n^2).
    #
    # This mattered. Geyer's rule below truncates at lag ~23 on a typical
    # chain, but the loop computed ALL n-1 lags first: at n = 200,000 that is
    # 199,999 dot products to consume 23, roughly 8,700x wasted work. It
    # finished on a quiet box and blew the per-test timeout under six xdist
    # workers, where pytest-timeout reported the traceback at whatever line
    # the thread happened to be on -- the assertion -- which read as a wrong
    # answer rather than a slow one.
    #
    # Zero-padding to at least 2n makes the circular correlation equal the
    # linear one, so this is the same quantity the loop produced, not an
    # approximation of it.
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(dev, n=m)
    acov = np.fft.irfft(f * np.conj(f), n=m)[: cap + 1].real
    rho = acov / denom
    rho[0] = 1.0

    # Geyer's initial positive sequence: truncate at the first non-positive
    # Gamma_m = rho_{2m} + rho_{2m+1}.
    total = 0.0
    K = 0
    m = 1
    while 2 * m + 1 <= cap:
        gamma = rho[2 * m] + rho[2 * m + 1]
        if gamma <= 0:
            break
        total += gamma
        K = 2 * m + 1
        m += 1
    # rho_1 sits outside the pairing and is always included.
    if cap >= 1:
        total += rho[1]
        K = max(K, 1)
    ess = n / (1.0 + 2.0 * total)
    return RichResult(
        payload={
            "estimate": float(ess),
            "ess": float(ess),
            "acf": rho[: K + 1],
            "sum_rho": float(total),
            "truncation_lag": int(K),
            "efficiency": float(ess / n),
            "n": int(n),
            "method": "ESS via Geyer's initial positive sequence (Geyer 1992)",
        }
    )


def cheatsheet():
    return "bayauto: ACF + ESS = N/(1 + 2 sum rho_k), Geyer initial positive sequence (Geyer 1992)."
