# morie.fn -- function file (rootcoder007/morie)
"""Shared ARE table and efficacy machinery for the Gibbons Ch. 13
modules.

Table 13.3.1 (PDF-verified, printed p. 492): ARE values of the
Wilcoxon signed-rank test (T+) and the sign test (K) against the
Student t test (T*), and of the sign test against the signed-rank
test, for four symmetric distributions.
"""

from . import _array_core as np
from ._sci_core import integrate

__all__ = ["ARE_TABLE", "efficacy_are"]

_PI = np.pi

# distribution -> {comparison: exact value}; Table 13.3.1
ARE_TABLE = {
    "uniform": {
        "wilcoxon_vs_t": 1.0, "sign_vs_t": 1.0 / 3.0, "sign_vs_wilcoxon": 1.0 / 3.0,
    },
    "normal": {
        "wilcoxon_vs_t": 3.0 / _PI, "sign_vs_t": 2.0 / _PI,
        "sign_vs_wilcoxon": 2.0 / 3.0,
    },
    "logistic": {
        "wilcoxon_vs_t": _PI**2 / 9.0, "sign_vs_t": _PI**2 / 12.0,
        "sign_vs_wilcoxon": 3.0 / 4.0,
    },
    "double_exponential": {
        "wilcoxon_vs_t": 1.5, "sign_vs_t": 2.0, "sign_vs_wilcoxon": 4.0 / 3.0,
    },
}

# Hodges-Lehmann (1956) bounds quoted on p. 492-493
HL_WILCOXON_LOWER_BOUND = 0.864  # inf over continuous symmetric F
HL_SIGN_LOWER_BOUND = 1.0 / 3.0  # inf over continuous unimodal symmetric F

# scale problem, Sec. 13.3.3 (PDF-verified): ARE(Mood, F | normal)
ARE_MOOD_VS_F_NORMAL = 15.0 / (2.0 * _PI**2)
# Klotz (1962) normal-scores scale test attains full efficiency at the normal
ARE_KLOTZ_VS_F_NORMAL = 1.0


def efficacy_are(f, cdf=None):
    r"""General location AREs from a density via the efficacy route.

    .. math::

       \mathrm{ARE}(T^+, t) &= 12\sigma^2\Big(\int f^2\Big)^2 \\
       \mathrm{ARE}(K, t)   &= 4\sigma^2 f(0)^2 \\
       \mathrm{ARE}(K, T^+) &= \frac{f(0)^2}{3\big(\int f^2\big)^2}

    for a density f symmetric about 0 with variance sigma^2
    (Gibbons Ch. 13.3, the efficacy ratios of the three tests).
    """
    m2, _ = integrate.quad(lambda x: x**2 * f(x), -np.inf, np.inf, limit=200)
    if m2 <= 0:
        raise ValueError("density must have positive variance.")
    if2, _ = integrate.quad(lambda x: f(x) ** 2, -np.inf, np.inf, limit=200)
    f0 = float(f(0.0))
    return {
        "wilcoxon_vs_t": float(12.0 * m2 * if2**2),
        "sign_vs_t": float(4.0 * m2 * f0**2),
        "sign_vs_wilcoxon": float(f0**2 / (3.0 * if2**2)),
        "sigma2": float(m2), "int_f2": float(if2), "f0": f0,
    }


def cheatsheet():
    return "_gb_are: Table 13.3.1 exact values + efficacy integrals from any density"
