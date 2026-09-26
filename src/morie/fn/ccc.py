# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Lin's concordance correlation coefficient."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as sp_stats
from ._containers import ESRes


def concordance_corr(x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> ESRes:
    r"""Lin's concordance correlation coefficient (CCC) with its z-transform interval.

    Lin (1989), as Hedderich, Sachs & Reynarowych (2023, Sec. 6.16.4) give it:

    .. math::

        \hat\rho_c = \frac{2 s_{xy}}{s_x^2 + s_y^2 + (\bar x - \bar y)^2},

    with the moments on divisor ``n``. The interval is Lin's: on
    :math:`z = \tanh^{-1}\hat\rho_c`,

    .. math::

        \mathrm{se}^2 = \frac{1}{n-2}\left[\frac{(1-r^2)\hat\rho_c^2(1-\hat\rho_c^2)}{r^2}
        + \frac{2\hat\rho_c^3(1-\hat\rho_c)u^2}{r}
        - \frac{\hat\rho_c^4u^4}{2r^2}\right] \big/ (1-\hat\rho_c^2)^2,

    :math:`u = (\bar y - \bar x)/(s_x^2 s_y^2)^{1/4}` and r the Pearson
    correlation, back-transformed by tanh -- DescTools' ``CCC(ci =
    "z-transform")``. At :math:`|\\hat\\rho_c| = 1` the interval is the point
    itself.

    Parameters
    ----------
    x, y : array-like
        Paired measurements.
    confidence : float
        Interval level.

    Returns
    -------
    ESRes
        ``estimate``, ``ci_lower``, ``ci_upper``, ``se`` (on the z scale);
        ``extra`` holds ``pearson_r``, the asymptotic interval, the scale
        shift ``v = s_y / s_x``, the location shift ``u`` and the bias
        correction factor ``C_b = rho_c / r``.

    References
    ----------
    Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics 45, 255-268; correction (2000) Biometrics 56,
    324-325. Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied
    Statistics: Methods Using R. Sec. 6.16.4.
    """
    xs = [float(v) for v in np.asarray(x, dtype=float).ravel()]
    ys = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    pairs = [(a, b) for a, b in zip(xs, ys) if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 3:
        raise ValueError("Need >= 3 paired observations.")
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sx2 = sum((a - mx) ** 2 for a in xs) / n
    sy2 = sum((b - my) ** 2 for b in ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in pairs) / n
    p = 2.0 * sxy / (sx2 + sy2 + (my - mx) ** 2)
    r = sxy / math.sqrt(sx2 * sy2)
    u = (my - mx) / (sx2 * sy2) ** 0.25
    sep = math.sqrt(
        max(
            0.0,
            (1 - r * r) * p * p * (1 - p * p) / (r * r) + 2 * p**3 * (1 - p) * u * u / r - 0.5 * p**4 * u**4 / (r * r),
        )
        / (n - 2)
    )
    zq = float(sp_stats.norm.ppf(1 - (1 - confidence) / 2))
    if abs(p) >= 1.0:
        # perfect (dis)agreement: the z-transform is infinite, the interval degenerate
        p = math.copysign(1.0, p)
        lo = hi = p
        se_t = 0.0
    else:
        t = math.atanh(p)
        se_t = sep / (1 - p * p)
        lo, hi = math.tanh(t - zq * se_t), math.tanh(t + zq * se_t)
    return ESRes(
        measure="concordance_corr",
        estimate=float(p),
        ci_lower=float(lo),
        ci_upper=float(hi),
        se=float(se_t),
        n=n,
        extra={
            "pearson_r": float(r),
            "asymptotic_ci": (float(p - zq * sep), float(p + zq * sep)),
            "scale_shift": math.sqrt(sy2 / sx2),
            "location_shift": float(u),
            "C_b": float(p / r),
        },
    )


ccc = concordance_corr


def cheatsheet() -> str:
    return "concordance_corr({}) -> Lin's concordance correlation coefficient."
