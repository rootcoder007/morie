# morie.fn -- function file (rootcoder007/morie)
"""Noncausal least-squares second derivative used to detect the dicrotic notch."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch4_dicrotic_notch_second_derivative"]

_COEF = np.array([2.0, -1.0, -2.0, -1.0, 2.0])  # taps for y(n-2)..y(n+2)


def rangayyan_ch4_dicrotic_notch_second_derivative(y, causal=False):
    r"""Lehner-Rangayyan least-squares second derivative.

    .. math:: p(n) = 2y(n-2) - y(n-1) - 2y(n) - y(n+1) + 2y(n+2)

    The five-tap least-squares estimate of the second derivative of the
    carotid pulse. It is deliberately noncausal (it looks two samples
    ahead); the book notes it "may be made causal by applying a delay
    of two samples", which ``causal=True`` does. The second derivative
    removes the constant downward slope of the carotid pulse and leaves
    the dicrotic notch standing out.

    Parameters
    ----------
    y : array-like, shape (n,)
        Carotid pulse signal, n >= 5.
    causal : bool, default False
        Delay the output by two samples so that ``p[n]`` depends only
        on ``y[..n]``.

    Returns
    -------
    RichResult
        keys: ``p`` (n,, zero-padded at the unusable ends), ``valid``
        (slice of fully-supported indices), ``coefficients``,
        ``causal``, ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Eq. (4.22), p. 228 (Sec. 4.3.5, detection of the
    dicrotic notch; after Lehner & Rangayyan).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 5:
        raise ValueError(f"need at least 5 samples for the 5-tap estimate, got {n}.")

    core = np.convolve(y, _COEF[::-1], mode="valid")  # p at indices 2 .. n-3
    p = np.zeros(n)
    if causal:
        p[4:] = core  # two-sample delay: p[n] uses y[n-4..n]
        valid = slice(4, n)
    else:
        p[2 : n - 2] = core
        valid = slice(2, n - 2)

    return RichResult(
        payload={
            "p": p,
            "valid": valid,
            "coefficients": _COEF.copy(),
            "causal": bool(causal),
            "n": int(n),
            "method": "Lehner-Rangayyan LS second derivative (Rangayyan Eq. 4.22, p. 228)",
        }
    )


def cheatsheet():
    return "rng196: p(n) = 2y(n-2) - y(n-1) - 2y(n) - y(n+1) + 2y(n+2) (Rangayyan Eq 4.22)"
