# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gibbs sampler (bivariate normal conditionals)."""

from . import _array_core as np

from ._richresult import RichResult
from .wsmnpb import _lcg_uniforms

__all__ = ["wasserman_gibbs_sampler"]


def _norm_inv(u):
    """Acklam's rational approximation of the standard normal
    quantile, |error| < 1.15e-9 — deterministic and portable."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if u < plow:
        q = np.sqrt(-2 * np.log(u))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if u > phigh:
        q = np.sqrt(-2 * np.log(1 - u))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = u - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def wasserman_gibbs_sampler(target, x0, n, seed=13):
    """
    Gibbs sampler for a bivariate normal with correlation rho.

    Formula: x_i^{t+1} ~ p(x_i | x_{-i}^t). For the standard
    bivariate normal the full conditionals are
    X | Y = y ~ N(rho y, 1 - rho^2) (and symmetrically), sampled by
    inversion on the shared exact-integer LCG so the whole chain is
    deterministic and language-portable. ``target`` is rho (a float
    in (-1, 1)) — the canonical Ch 24 example; arbitrary targets
    belong to Metropolis (wsmmcm).

    Parameters
    ----------
    target : float
        Correlation rho of the bivariate normal, |rho| < 1.
    x0 : sequence of 2 floats
        Starting point (x, y).
    n : int
        Sweeps (each updates both coordinates), >= 1.
    seed : int
        LCG seed.

    Returns
    -------
    result : dict
        Keys: estimate (sample correlation of the chain), samples_x,
        samples_y, mean_x, mean_y, n, method.

    References
    ----------
    Wasserman (2004), Ch 24, section 24.4 (Gibbs sampling).

    Examples
    --------
    >>> out = wasserman_gibbs_sampler(0.9, [0.0, 0.0], 4000)
    >>> 0.85 < out["estimate"] < 0.95
    True
    >>> abs(out["mean_x"]) < 0.1
    True
    >>> wasserman_gibbs_sampler(1.0, [0.0, 0.0], 100)
    Traceback (most recent call last):
        ...
    ValueError: the correlation must satisfy |rho| < 1; got 1.0.
    """
    rho = float(target)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"the correlation must satisfy |rho| < 1; got {rho}.")
    n = int(n)
    if n < 1:
        raise ValueError(f"the sampler needs n >= 1 sweeps; got {n}.")
    x, y = (float(v) for v in x0)
    s = np.sqrt(1.0 - rho * rho)
    u = _lcg_uniforms(2 * n, seed)
    xs, ys = [], []
    for t in range(n):
        x = rho * y + s * _norm_inv(u[2 * t])
        y = rho * x + s * _norm_inv(u[2 * t + 1])
        xs.append(x)
        ys.append(y)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    corr = float(np.corrcoef(xs, ys)[0, 1])
    return RichResult(payload={
        "estimate": corr, "samples_x": [float(v) for v in xs],
        "samples_y": [float(v) for v in ys],
        "mean_x": float(np.mean(xs)), "mean_y": float(np.mean(ys)),
        "n": n,
        "method": "Gibbs on bivariate normal conditionals, LCG inversion"})


def cheatsheet():
    return "wsmgib: X|y ~ N(rho y, 1-rho^2) alternating; LCG + Acklam inversion"
