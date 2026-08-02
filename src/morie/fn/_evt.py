# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the extreme-value shelf.

Two recurring objects. The EXTREME-VALUE INDEX xi (the shape): the
GEV shape for block maxima, the GPD shape for threshold excesses,
and 1/alpha for a regularly varying tail -- one parameter, three
guises, and the estimators here (Hill, Pickands,
Dekkers-Einmahl-de Haan, the PWM fits) all target it. And the
EXTREMAL INDEX theta in (0, 1]: the reciprocal mean cluster size of
exceedances in a stationary series, the number that separates
"the data are heavy-tailed" from "the exceedances arrive in
clumps".

Probability-weighted moments (Greenwood et al. 1979) are the
workhorse:

    b_r = n^-1 sum_j [ (j-1)(j-2)...(j-r) /
                       ((n-1)(n-2)...(n-r)) ] x_(j)

on the ascending order statistics -- the unbiased estimator, not the
plotting-position approximation, because the difference is exactly
the kind of small-sample bias these methods exist to avoid.
L-moments (Hosking 1990) are the linear combinations
l1 = b0, l2 = 2 b1 - b0, l3 = 6 b2 - 6 b1 + b0, and their GEV/GPD
inversions have closed forms the tests verify against.
"""

from . import _array_core as np

__all__ = ["pwm_b", "l_moments", "gev_from_lmoments", "gpd_from_pwm",
           "top_order", "EULER_GAMMA"]

EULER_GAMMA = 0.5772156649015329


def pwm_b(x, r):
    r"""The unbiased probability-weighted moment
    :math:`b_r = E[X F(X)^r]`'s sample version (Greenwood et al.
    1979; Hosking, Wallis and Wood 1985, Eq. (4))."""
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.size
    if n < r + 1:
        raise ValueError(f"b_{r} needs at least {r + 1} observations.")
    j = np.arange(1, n + 1, dtype=float)
    w = np.ones(n)
    for k in range(1, r + 1):
        w *= (j - k) / (n - k)
    return float(np.mean(w * xs))


def l_moments(x):
    """The first three L-moments l1, l2, l3 and the L-skewness t3."""
    b0 = pwm_b(x, 0)
    b1 = pwm_b(x, 1)
    b2 = pwm_b(x, 2)
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    if l2 == 0:
        raise ValueError("the second L-moment is zero; the data are "
                         "constant.")
    return l1, l2, l3, l3 / l2


def gev_from_lmoments(l1, l2, t3):
    r"""Hosking's (1990) L-moment inversion for the GEV, using his
    approximation (also Hosking, Wallis and Wood 1985, Eq. (14))

    .. math:: k \approx 7.8590\,c + 2.9554\,c^2, \qquad
              c = \frac{2}{3 + t_3} - \frac{\log 2}{\log 3},

    then exactly

    .. math:: \alpha = \frac{l_2 k}{(1 - 2^{-k})\Gamma(1 + k)},
              \qquad \mu = l_1 - \frac{\alpha}{k}
              \{1 - \Gamma(1 + k)\}.

    Sign convention: Hosking's ``k`` is MINUS the extreme-value index
    xi. Heavy tails mean k < 0 and xi > 0; conflating the two flips
    every Frechet into a Weibull, so both are returned and named.
    """
    from ._sci_core import gamma as G

    c = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c ** 2
    if abs(k) < 1e-9:
        # Gumbel limit
        alpha = l2 / np.log(2)
        mu = l1 - EULER_GAMMA * alpha
        return mu, alpha, 0.0
    alpha = l2 * k / ((1 - 2.0 ** (-k)) * G(1 + k))
    mu = l1 - alpha / k * (1 - G(1 + k))
    return float(mu), float(alpha), float(k)


def gpd_from_pwm(x):
    r"""Hosking and Wallis (1987), Eq. (13)-(14): with
    :math:`a_0 = b_0` and :math:`a_1 = b_0 - 2 b_1` ... equivalently
    via the first two L-moments of excesses,

    .. math:: k = \frac{l_1}{l_2} - 2, \qquad
              \sigma = l_1 (1 + k) ... \text{in Hosking's k}:
              \hat k = l_1/l_2 - 2,\ \hat\sigma = (1 + \hat k) l_1
              \cdot \frac{l_2}{l_1} \cdot ... 

    Written plainly: :math:`\hat k = l_1/l_2 - 2` and
    :math:`\hat\sigma = (1 + \hat k)\,l_2\,(l_1/l_2)` reduces to
    :math:`\hat\sigma = l_1 (1 + \hat k)`. Hosking's k is minus the
    GPD shape xi.
    """
    b0 = pwm_b(x, 0)
    b1 = pwm_b(x, 1)
    l1 = b0
    l2 = 2 * b1 - b0
    if l2 <= 0:
        raise ValueError("the excesses' second L-moment must be positive.")
    k = l1 / l2 - 2.0
    sigma = l1 * (1.0 + k)
    return float(sigma), float(k)


def top_order(x, k):
    """The k+1 largest order statistics, descending."""
    xs = np.sort(np.asarray(x, dtype=float).ravel())[::-1]
    k = int(k)
    if not 1 <= k < xs.size:
        raise ValueError(f"k must lie in 1..{xs.size - 1}, got {k}.")
    return xs[:k + 1]


def cheatsheet():
    return ("_evt: Hosking's k is MINUS the extreme-value index xi -- "
            "heavy tail means k < 0, xi > 0")
