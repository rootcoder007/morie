# morie.fn -- function file (rootcoder007/morie)
"""Black-Scholes implied volatility by root-finding."""

from . import _array_core as np
from ._sci_core import optimize
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["vol_implied_volatility_bs"]


def _bs_price(S, K, T, r, sigma, kind):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def vol_implied_volatility_bs(S, K, T, r, price_obs, kind="call"):
    r"""Invert Black-Scholes for the implied volatility.

    Solves :math:`C_{BS}(\sigma) = C_{obs}` by Brent's method on
    :math:`\sigma \in [10^{-4}, 5]`. The observed price must respect
    the no-arbitrage bounds (intrinsic value below, forward bound
    above); a price outside them has no implied volatility and the
    error says which bound failed.

    Parameters
    ----------
    S, K, T, r : float
        Spot, strike, time to expiry (years, > 0), risk-free rate.
    price_obs : float
        Observed option price.
    kind : {"call", "put"}, default "call"

    Returns
    -------
    RichResult
        keys: ``implied_vol``, ``price_check`` (BS price at the
        root), ``vega`` (at the root), ``kind``, ``method``.

    References
    ----------
    Black, F. & Scholes, M. (1973). The pricing of options and
    corporate liabilities. *Journal of Political Economy*, 81(3),
    637-654.
    """
    S, K, T, r, price_obs = (float(v) for v in (S, K, T, r, price_obs))
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}.")

    disc = np.exp(-r * T)
    if kind == "call":
        lo_b, hi_b = max(S - K * disc, 0.0), S
    else:
        lo_b, hi_b = max(K * disc - S, 0.0), K * disc
    if not lo_b < price_obs < hi_b:
        raise ValueError(
            f"price {price_obs} violates the no-arbitrage bounds ({lo_b:.6g}, {hi_b:.6g})."
        )

    f = lambda s: _bs_price(S, K, T, r, s, kind) - price_obs
    sigma = float(optimize.brentq(f, 1e-4, 5.0, xtol=1e-12))
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    vega = float(S * stats.norm.pdf(d1) * np.sqrt(T))

    return RichResult(
        payload={
            "implied_vol": sigma,
            "price_check": float(_bs_price(S, K, T, r, sigma, kind)),
            "vega": vega,
            "kind": kind,
            "method": "Black-Scholes implied volatility (Brent root on [1e-4, 5])",
        }
    )


def cheatsheet():
    return "volopn: brentq on C_BS(sigma) = C_obs after the no-arbitrage bound check"
