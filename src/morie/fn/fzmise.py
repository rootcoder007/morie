# morie.fn -- function file (rootcoder007/morie)
"""MISE of kernel density estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_mise", "fauzi_mise_kdfe"]


def fauzi_mise(n, h=None, R_K=None, mu2_K=1.0, R_f2=1.0, sigma=1.0):
    r"""Mean integrated squared error of the kernel density estimator
    (Fauzi Ch. 1):

    .. math:: \mathrm{MISE} \approx \frac{R(K)}{nh}
              + \frac{h^4}{4}\mu_2(K)^2 R(f''),

    variance falling in h and squared bias rising in it.

    Minimising gives the classical

    .. math:: h_{opt} = \left[\frac{R(K)}
              {n\,\mu_2(K)^2 R(f'')}\right]^{1/5}
              \propto n^{-1/5},
              \qquad
              \mathrm{MISE}_{opt} \propto n^{-4/5},

    and that :math:`n^{-4/5}` is the ceiling for a second-order
    kernel and a twice-differentiable density -- strictly worse than
    the parametric :math:`n^{-1}`, and not improvable without either
    more smoothness or a higher-order kernel. Both the optimum and
    the attainable rate are returned, so the gap to parametric is a
    number rather than a remark.

    Parameters
    ----------
    n : int
        Sample size.
    h : float, optional
        Bandwidth to evaluate MISE at; the optimum otherwise.
    R_K : float, optional
        :math:`\int K^2`; the Gaussian value otherwise.
    mu2_K : float
        :math:`\int u^2 K(u)du`.
    R_f2 : float
        :math:`\int (f'')^2`, the roughness of the truth.
    sigma : float
        Scale, for the normal reference rule.

    Returns
    -------
    RichResult
        keys: ``mise``, ``variance_part``, ``bias_part``, ``h``,
        ``h_optimal``, ``mise_optimal``, ``rate_exponent`` (-4/5),
        ``parametric_rate_exponent`` (-1), ``n``, ``method``.
    """
    nn = int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    rk = 1.0 / (2.0 * np.sqrt(np.pi)) if R_K is None else float(R_K)
    m2 = float(mu2_K)
    rf = float(R_f2)
    if rk <= 0 or m2 <= 0 or rf <= 0:
        raise ValueError("R_K, mu2_K and R_f2 must all be positive.")
    h_opt = float((rk / (nn * m2 ** 2 * rf)) ** 0.2)
    hh = h_opt if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    var = rk / (nn * hh)
    bias = hh ** 4 / 4.0 * m2 ** 2 * rf
    return RichResult(payload={
        "mise": float(var + bias), "variance_part": float(var),
        "bias_part": float(bias), "h": hh, "h_optimal": h_opt,
        "mise_optimal": float(rk / (nn * h_opt) + h_opt ** 4 / 4 * m2 ** 2 * rf),
        "rate_exponent": -0.8, "parametric_rate_exponent": -1.0,
        "bandwidth_rate": "h_opt proportional to n^{-1/5}",
        "ceiling_note": "n^{-4/5} is the best a second-order kernel can do "
                        "for a twice-differentiable density",
        "n": nn,
        "method": "MISE = R(K)/(nh) + h^4 mu2^2 R(f'')/4; the two terms pull opposite ways"})


def cheatsheet():
    return "fzmise: h ~ n^{-1/5} gives MISE ~ n^{-4/5} -- the ceiling, and short of parametric n^{-1}"


#: Catalogue alias for :func:`fauzi_mise`.
fauzi_mise_kdfe = fauzi_mise


# compact alias per ledger/NAMING.md
fauzimise = fauzi_mise


# compact alias per ledger/NAMING.md
fauzimisekdfe = fauzi_mise_kdfe
