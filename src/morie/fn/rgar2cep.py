# morie.fn -- function file (rootcoder007/morie)
"""Cepstral coefficients from AR coefficients."""

import math as _math

from ._richresult import RichResult

__all__ = ["rangayyan_ar_to_cepstrum", "ar2cep"]


def rangayyan_ar_to_cepstrum(a_coeffs, gain=None):
    r"""AR coefficients to complex cepstrum, Rangayyan eq. (7.65).

    .. math::
        \hat h(1) &= -a_1 \\
        \hat h(n) &= -a_n - \sum_{k=1}^{n-1}
                     \left(1 - \frac{k}{n}\right) a_k \hat h(n-k),
                     \quad 1 < n \le P

    Derived by expanding :math:`\ln H(z)` as a Laurent series
    (eq. 7.61-7.64) and equating like powers of :math:`z^{-1}`.

    The alternative spelling :math:`\sum (k/n)\,\hat h(k)\,a_{n-k}` is
    the SAME recursion reindexed: put :math:`j = n - k` and
    :math:`(1 - k/n)\,a_k\,\hat h(n-k)` becomes
    :math:`(j/n)\,\hat h(j)\,a_{n-j}`.  The book's form is used here
    because it is the one the citation points at.

    Going through the AR coefficients avoids the phase unwrapping that
    the FFT-based cepstrum needs (Section 4.7.3), which is the practical
    reason to prefer this route.

    Parameters
    ----------
    a_coeffs : sequence
        AR coefficients :math:`a_1 \dots a_P` in the sign convention of
        :math:`A(z) = 1 + \sum a_k z^{-k}` -- the residual filter, so
        that a stable model has its poles inside the unit circle.
    gain : float, optional
        Model gain :math:`G`.  When given, ``c0 = log(G)`` is returned as
        the zeroth cepstral coefficient; the recursion itself does not
        involve it.

    Returns
    -------
    RichResult
        ``cepstrum`` (n = 1..P), ``c0`` (or None).

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd ed.
    Wiley-IEEE Press, Section 7.5.3, eqs. (7.61)-(7.65).
    """
    a = [float(v) for v in a_coeffs]
    p = len(a)
    if p == 0:
        raise ValueError("need at least one AR coefficient")

    h = [0.0] * (p + 1)          # h[n] for n = 1..P
    for n in range(1, p + 1):
        acc = -a[n - 1]
        for k in range(1, n):
            acc -= (1.0 - k / n) * a[k - 1] * h[n - k]
        h[n] = acc
    cep = h[1:]

    c0 = None
    if gain is not None:
        g = float(gain)
        if g <= 0:
            raise ValueError("gain must be positive")
        c0 = _math.log(g)

    return RichResult(
        title="AR to cepstrum (Rangayyan eq. 7.65)",
        summary_lines=[("order", p)],
        payload={"cepstrum": cep, "c0": c0, "order": p,
                 "method": "Rangayyan (2024) eq. (7.65)"},
    )


ar2cep = rangayyan_ar_to_cepstrum


def cheatsheet():
    return "ar2cep: h(n) = -a_n - sum (1 - k/n) a_k h(n-k), eq 7.65"
