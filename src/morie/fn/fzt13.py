# morie.fn -- function file (rootcoder007/morie)
"""Bias of the modified gamma kernel density estimator (Theorem 1.3)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mgkbias", "fauzi_thm1_3_mise_mgkde"]


def mgkbias(x, h, f, fp, fpp, fppp, book=True):
    r"""Bias of the modified gamma kernel density estimator (Theorem 1.3).

    Theorem 1.3, Eq. (1.15):

    .. math:: \mathrm{Bias}[\tilde f_X(x)] = -2\Big(b(x) -
              \frac{a^2(x)}{2f_X(x)}\Big)h + o(h) + O(n^{-1}h^{-1/4}),

    with :math:`a` and :math:`b` from (1.16) and (1.17). The order is back
    to :math:`h` -- the same as Chen's -- while the variance keeps the
    smaller order of Theorem 1.1, which is the point of the whole chapter.

    Two notes on (1.15)-(1.17), stated plainly.

    (i) In the extracted text of (1.15) the superscript on :math:`a` is
    lost; the proof of Theorem 1.2 writes :math:`a^2(x)/(2f_X(x))` three
    times, and Eq. (2.8) of Fauzi's doctoral thesis (below) prints the
    square, so it is certain.

    (ii) (1.17) prints ``b(x) = x + f''(x)/2 + x^2(x/3 + 1/2) f^(3)(x)``,
    with a bare ADDITIVE ``x``. That reading is confirmed against the
    primary source, where the prime marks are legible in the text layer:
    Fauzi, R. R. (2020), *Bias Reduction of Kernel-Type Estimators without
    Boundary Problems*, doctoral thesis, Kyushu University (institutional
    repository, ``math0257``), Eq. (2.10). So it is what both sources
    print, not a transcription artefact.

    It is nevertheless dimensionally impossible. ``b(x)`` multiplies ``h``
    to give a density, so ``b`` carries the units of :math:`f''`; a bare
    ``x`` carries units of length and cannot be added to it. Carrying the
    book's own Taylor argument through
    (:math:`\mu_W = x+\sqrt h`,
    :math:`\mathrm{Var}(W)=\sqrt h(x+\sqrt h)^2`,
    :math:`E(W-\mu_W)^3 = 2h(x+\sqrt h)^3`) gives the ``h`` coefficient as
    :math:`(x + 1/2)f''(x) + x^2(x/3+1/2)f^{(3)}(x)` -- the ``x``
    MULTIPLYING :math:`f''`, which is dimensionally consistent and differs
    from the printed form only by a missing pair of brackets.

    ``book=True`` (the default) reproduces exactly what both published
    sources print. ``book=False`` uses the derived form. They agree only
    when :math:`f''(x) = 1`.

    Parameters
    ----------
    x : float
        Evaluation point, ``x >= 0``.
    h : float
        Bandwidth.
    f, fp, fpp, fppp : float
        ``f_X(x)`` and its first three derivatives at ``x``.
    book : bool, default True
        Reproduce (1.17) as printed; ``False`` uses the derived form.

    Returns
    -------
    RichResult
        Keys ``bias``, ``a``, ``b``, ``h``, ``book``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 1.3, Eqs. (1.15)-(1.17).
    """
    x = float(x)
    h = float(h)
    f = float(f)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if f == 0.0:
        raise ValueError("(1.15) divides by f_X(x); it must be non-zero.")
    a = float(fp) + 0.5 * x * x * float(fpp)
    if book:
        b = x + 0.5 * float(fpp) + x * x * (x / 3.0 + 0.5) * float(fppp)
    else:
        b = (x + 0.5) * float(fpp) + x * x * (x / 3.0 + 0.5) * float(fppp)
    bias = -2.0 * (b - a * a / (2.0 * f)) * h
    return RichResult(
        payload={
            "bias": float(bias),
            "a": float(a),
            "b": float(b),
            "h": h,
            "book": bool(book),
            "method": "modified gamma KDE bias (Theorem 1.3)",
        }
    )


fauzi_thm1_3_mise_mgkde = mgkbias


def cheatsheet():
    return "fzt13: modified gamma KDE bias is O(h) again, coefficient -2(b - a^2/2f) (Thm 1.3)"


# CANONICAL TEST
# >>> r = mgkbias(x=1.0, h=0.01, f=0.5, fp=0.0, fpp=0.0, fppp=0.0)
# >>> abs(r['bias'] - (-2 * 1.0 * 0.01)) < 1e-15
# True
