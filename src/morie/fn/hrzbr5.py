# morie.fn -- function file (rootcoder007/morie)
"""Higher-order kernels: the bias-reduction device of deconvolution.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer.  Section 5.2.4, page 153 (volume
[Pages 135-188], read as a rendered page image) is the bias-reduction
section this function is filed under; the correction printed there is
the explicit smoothing correction

    f_hat_n_eps(z) = f_n_eps(z) - (1/2) v_n_eps^2 f_n_eps(z) sigma_zeta^2   (5.39)

with f_n_eps the second derivative, which removes the O(v^2) term by
subtracting an estimate of it.  The device this function implements is
the other one named in its own specification and used throughout the
appendix (Section A.1, p. 236, volume [Pages 233-255]): a kernel of
order r, that is one satisfying

    integral K(u) du = 1,
    integral u^j K(u) du = 0   for j = 1, ..., r - 1,
    integral u^r K(u) du != 0,

which reduces the leading smoothing bias from O(h^2) to O(h^r).

The order-r Gaussian kernel is built here in the classical Hermite form

    K_r(u) = phi(u) sum_{j=0}^{r/2 - 1} (-1)^j / (2^j j!) He_{2j}(u),

phi the standard normal density and He the probabilists Hermite
polynomials.  r = 2 recovers phi itself, r = 4 gives phi(u)(3 - u^2)/2
and r = 6 gives phi(u)(15 - 10 u^2 + u^4)/8.  The moments returned are
computed exactly, by expanding the polynomial factor in powers of u and
using integral u^m phi(u) du = (m-1)!! for even m and 0 for odd m; no
quadrature is involved, so moments 1 through r-1 come back as exact
floating-point zeros.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["horowitz_bias_reduction_deconv"]


def _hermite_coeffs(m):
    """Coefficients of the probabilists Hermite polynomial He_m, low order first."""
    prev = [1.0]
    if m == 0:
        return prev
    cur = [0.0, 1.0]
    for k in range(1, m):
        nxt = [0.0] * (k + 2)
        for i in range(len(cur)):
            nxt[i + 1] += cur[i]
        for i in range(len(prev)):
            nxt[i] -= k * prev[i]
        prev = cur
        cur = nxt
    return cur


def _gauss_moment(m):
    """integral u^m phi(u) du: (m-1)!! for even m, 0 for odd m."""
    if m % 2 == 1:
        return 0.0
    v = 1.0
    k = m - 1
    while k > 1:
        v *= k
        k -= 2
    return v


def horowitz_bias_reduction_deconv(bandwidth, kernel_order):
    """Order-r Gaussian kernel, its exact moments, and the bias order it buys.

    Parameters
    ----------
    bandwidth : float
        The smoothing parameter h; only its bias order h^r is reported.
    kernel_order : int
        r, an even integer at least 2.

    Returns
    -------
    reduced_bias_estimate : h^r, the order of the leading smoothing bias
    coefficients : the polynomial factor of K_r, low order first
    moments : integral u^j K_r(u) du for j = 0, ..., r
    """
    h = float(bandwidth)
    r = int(kernel_order)
    if r < 2 or r % 2 != 0:
        raise ValueError("horowitz_bias_reduction_deconv: kernel_order must be an even integer >= 2")
    if h <= 0.0:
        raise ValueError("horowitz_bias_reduction_deconv: bandwidth must be positive")
    poly = [0.0] * (r - 1)
    for j in range(r // 2):
        w = ((-1.0) ** j) / ((2.0 ** j) * math.factorial(j))
        hc = _hermite_coeffs(2 * j)
        for i in range(len(hc)):
            poly[i] += w * hc[i]
    moments = []
    for k in range(r + 1):
        s = 0.0
        for i in range(len(poly)):
            if poly[i] != 0.0:
                s += poly[i] * _gauss_moment(k + i)
        moments.append(s)
    return RichResult(
        title="Higher-order kernel bias reduction",
        summary_lines=[("order", r), ("bandwidth", h)],
        payload={
            "estimate": h ** r,
            "reduced_bias_estimate": h ** r,
            "bias_order": r,
            "kernel_order": r,
            "bandwidth": h,
            "coefficients": poly,
            "moments": moments,
            "leading_moment": moments[r],
            "n": r,
            "method": "Horowitz (2009) Sec. A.1 p.236 order-r kernel; Hermite construction on the Gaussian",
        },
    )


def cheatsheet():
    return "hrzbr5: Bias reduction for deconvolution via higher-order kernels"
