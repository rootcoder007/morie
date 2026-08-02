# morie.fn -- function file (rootcoder007/morie)
"""MISE-optimal bandwidth."""

from . import _array_core as np

from ._horowitz import kernel
from ._richresult import RichResult

__all__ = ["hrz_bandwidth_optimal", "horowitz_optimal_bandwidth_kde"]


def hrz_bandwidth_optimal(x, kernel_name="gaussian", f_second_deriv_l2=None):
    r"""MISE-optimal bandwidth (Horowitz Ch. 2):

    .. math:: h_{opt} = \left[\frac{R(K)}
              {\mu_2(K)^2 \int (f'')^2\, n}\right]^{1/5},
              \qquad R(K) = \int K^2.

    The formula depends on the UNKNOWN :math:`\int (f'')^2`, which is
    why it cannot be used directly and every practical rule (Silverman,
    plug-in, cross-validation) is an approximation to it. When that
    functional is not supplied it is estimated from a normal
    reference, and the result says which route was taken.

    Parameters
    ----------
    x : array-like
        Sample.
    kernel_name : str
        Kernel.
    f_second_deriv_l2 : float, optional
        The true :math:`\int (f'')^2`, if known.

    Returns
    -------
    RichResult
        keys: ``h_opt``, ``R_K``, ``mu2_K``, ``f2_l2``,
        ``normal_reference_used`` (bool), ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (bandwidth selection).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations.")
    u = np.linspace(-10, 10, 20001)
    Ku = kernel(u, kernel_name)
    R_K = float(np.trapezoid(Ku**2, u))
    mu2 = float(np.trapezoid(u**2 * Ku, u))
    if mu2 <= 0:
        raise ValueError("kernel has non-positive second moment.")
    ref = f_second_deriv_l2 is None
    if ref:
        sd = float(x.std(ddof=1))
        if sd <= 0:
            raise ValueError("x has zero spread.")
        f2 = 3.0 / (8.0 * np.sqrt(np.pi) * sd**5)  # normal reference
    else:
        f2 = float(f_second_deriv_l2)
        if f2 <= 0:
            raise ValueError("f_second_deriv_l2 must be positive.")
    h = (R_K / (mu2**2 * f2 * n)) ** 0.2
    return RichResult(payload={"h_opt": float(h), "R_K": R_K, "mu2_K": mu2,
                               "f2_l2": f2, "normal_reference_used": bool(ref),
                               "n": int(n),
                               "method": "h_opt = [R(K)/(mu2^2 int(f'')^2 n)]^{1/5}"})


def cheatsheet():
    return "hrzbwopt: depends on unknown int(f'')^2 -- every practical rule approximates it"


#: Catalogue alias for :func:`hrz_bandwidth_optimal`.
horowitz_optimal_bandwidth_kde = hrz_bandwidth_optimal
