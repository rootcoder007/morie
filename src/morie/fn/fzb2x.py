# morie.fn -- function file (rootcoder007/morie)
"""The b_2 bias coefficient of the kernel distribution function estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfb2", "fauzi_b2_coefficient_kdfe"]


def kdfb2(fp, mu2=1.0):
    r"""The b_2 bias coefficient of the kernel distribution function estimator.

    Eq. (2.7):

    .. math:: b_2(x) = \frac{f_X'(x)}2\int_{-\infty}^{\infty}w^2K(w)\,dw.

    The leading bias coefficient of the KDFE: :math:`\mathrm{Bias}[\hat
    F_h(x)] = h^2 b_2(x)/F_X(x)\cdot F_X(x) = h^2 b_2(x) + o(h^2)`.

    Note which derivative appears. A kernel DENSITY estimator's leading
    bias carries :math:`f''`; the distribution-function estimator carries
    :math:`f'`, one order lower, because it smooths with the INTEGRATED
    kernel :math:`W`. That single fact is why the whole book's bandwidth
    rate is :math:`n^{-1/3}` and not :math:`n^{-1/5}`.

    ``mu2`` defaults to 1, the Gaussian value; pass ``mu2 = 0.2`` for the
    Epanechnikov kernel. It is an explicit argument, never estimated from
    the data -- the kernel is a modelling choice, not a random quantity.

    Parameters
    ----------
    fp : float
        ``f_X'(x)``.
    mu2 : float, default 1.0
        ``int w^2 K(w) dw``; 1 for the Gaussian kernel.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``mu2``, ``method``.


    Naming note. The backlog assigns this row the public name
    ``fauzi_b2_coefficient``, but that name is already taken -- by
    ``morie.fn.fzb2t``, for the Chapter 4 coefficient
    :math:`b_2(t)` of Eq. (4.15), which is a different quantity that
    merely shares the book's symbol. Binding it here as well would make
    the resolved function depend on import order, and on the R side would
    silently rebind the exported ``morie_fauzi_b2_coefficient``. The
    legacy spelling kept here is therefore
    ``fauzi_b2_coefficient_kdfe``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (2.7).
    """
    val = float(fp) / 2.0 * float(mu2)
    return RichResult(
        payload={
            "estimate": float(val),
            "mu2": float(mu2),
            "method": "b_2 bias coefficient of the KDFE (Eq. 2.7)",
        }
    )


fauzi_b2_coefficient_kdfe = kdfb2


def cheatsheet():
    return "fzb2x: b_2(x) = f'(x) mu_2(K) / 2 -- the KDFE's leading bias coefficient (Eq. 2.7)"


# CANONICAL TEST
# >>> kdfb2(fp=0.4)['estimate'] == 0.2
# True
