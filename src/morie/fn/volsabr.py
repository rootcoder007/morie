"""SABR implied volatility (Hagan, Kumar, Lesniewski & Woodward 2002)."""

import math

from ._richresult import RichResult

__all__ = ["volsabr", "sabr_implied_volatility"]


def volsabr(K, f, T, alpha, beta, rho, nu):
    """
    SABR model Black implied volatility, Hagan et al. (2002).

    Their Eq. 2.17a-c: with z = (nu/alpha) (f K)^((1-beta)/2)
    log(f/K) and x(z) = log[(sqrt(1 - 2 rho z + z^2) + z - rho) /
    (1 - rho)],

        sigma_B = alpha / [ (fK)^((1-beta)/2)
                    (1 + (1-beta)^2/24 log^2(f/K)
                       + (1-beta)^4/1920 log^4(f/K)) ]
                  * (z / x(z))
                  * { 1 + [ (1-beta)^2/24 alpha^2/(fK)^(1-beta)
                          + rho beta nu alpha / (4 (fK)^((1-beta)/2))
                          + (2 - 3 rho^2)/24 nu^2 ] T },

    reducing at the money (K = f) to their Eq. 2.18.  The z/x(z)
    factor tends to 1 as z -> 0 and is series-expanded for tiny z to
    avoid cancellation.

    Sources
    -------
    Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E.
    (2002). Managing smile risk. *Wilmott Magazine*, September,
    84-108, Eqs. 2.17a-2.17c and 2.18 (local copy
    fetched-wave3/hagan-2002-sabr-managing-smile-risk.pdf).

    Parameters
    ----------
    K, f : float
        Strike and forward (both > 0).
    T : float
        Time to exercise t_ex (years).
    alpha : float
        Instantaneous volatility parameter (> 0).
    beta : float
        CEV exponent in [0, 1].
    rho : float
        Correlation in (-1, 1).
    nu : float
        Vol-of-vol (>= 0).

    Returns
    -------
    RichResult
        Keys: estimate (sigma_B), z, x_z, atm (Eq. 2.18 value at
        this f).
    """
    K = float(K)
    f = float(f)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    nu = float(nu)
    if K <= 0 or f <= 0 or alpha <= 0 or T < 0:
        raise ValueError("K, f, alpha must be positive; T >= 0")
    if not (0.0 <= beta <= 1.0) or not (-1.0 < rho < 1.0) or nu < 0:
        raise ValueError("need 0 <= beta <= 1, -1 < rho < 1, nu >= 0")
    omb = 1.0 - beta
    lfk = math.log(f / K)
    fkb = (f * K) ** (omb / 2.0)
    denom = fkb * (1.0 + omb ** 2 / 24.0 * lfk ** 2
                   + omb ** 4 / 1920.0 * lfk ** 4)
    z = (nu / alpha) * fkb * lfk
    if abs(z) < 1e-7:
        # z/x(z) = 1 + rho z/2 + (2 - 3 rho^2) z^2 / 12 + O(z^3)
        zoxz = 1.0 + rho * z / 2.0 + (2.0 - 3.0 * rho ** 2) * z * z / 12.0
        x_z = z / zoxz if z != 0.0 else float("nan")
    else:
        x_z = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z)
                        + z - rho) / (1.0 - rho))
        zoxz = z / x_z
    corr = 1.0 + (omb ** 2 / 24.0 * alpha ** 2 / (f * K) ** omb
                  + rho * beta * nu * alpha / (4.0 * fkb)
                  + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2) * T
    sigma = alpha / denom * zoxz * corr
    atm = alpha / f ** omb * (
        1.0 + (omb ** 2 / 24.0 * alpha ** 2 / f ** (2.0 * omb)
               + rho * beta * alpha * nu / (4.0 * f ** omb)
               + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2) * T)
    return RichResult(payload={
        "estimate": sigma,
        "z": z,
        "x_z": x_z,
        "atm": atm,
        "method": "SABR implied vol (Hagan et al. 2002, Eq. 2.17)",
    })


# long descriptive alias (stub-era name)
sabr_implied_volatility = volsabr


def cheatsheet():
    return "volsabr: sigma_B = alpha/((fK)^((1-b)/2)(1+...)) * z/x(z) * (1+corr T)"

# public names resolved by fn/_lazy_map.json
vol_sabr_implied = volsabr
volsabrimplied = volsabr
