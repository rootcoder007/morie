# SPDX-License-Identifier: AGPL-3.0-or-later
"""The interpolation step of a type-7 quantile, as R's quantile(type = 7) takes it."""

import math


def lerp7(a: float, b: float, g: float) -> float:
    """``a + g * (b - a)`` for the order statistics ``a <= b`` at weight ``g`` on ``b``.

    The lower value alone when ``g`` is 0, and the weighted mean when ``b - a`` is not finite
    (an infinite value, or two finite ones whose difference overflows), where the plain form
    gave NaN (``0 * inf``, ``inf - inf``) or an infinity; otherwise the plain form, bit for bit.
    The same rule as ``quantile7_sorted`` in rmoriebricklayer 1.1.1.

    Examples:
        >>> lerp7(2.0, math.inf, 0.0), lerp7(-math.inf, 1.0, 0.5), lerp7(-1e308, 1e308, 0.5)
        (2.0, -inf, 0.0)
    """
    if g == 0:
        return a
    d = b - a
    if math.isfinite(d):
        return a + g * d
    return (1.0 - g) * a + g * b
