# morie.fn -- function file (rootcoder007/morie)
"""Rangayyan Ch. 4 synthetic three-event test signal (Eq. 4.51)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch4_test_signal_three_events"]


def rangayyan_ch4_test_signal_three_events(n=36):
    r"""Synthetic signal of three scaled repetitions of one basic pattern.

    .. math::

        x(n) = 3\delta(n-5) + 2\delta(n-6) + \delta(n-7)
             + 1.5\delta(n-16) + \delta(n-17) + 0.5\delta(n-18)
             + 0.75\delta(n-26) + 0.5\delta(n-27) + 0.25\delta(n-28)

    (Eq. 4.51), equivalently :math:`x(n) = g(n-5) + 0.5\,g(n-16) +
    0.25\,g(n-26)` with the basic pattern :math:`g(n) = 3\delta(n) +
    2\delta(n-1) + \delta(n-2)` (Eqs. 4.52-4.53). The book uses it to
    illustrate matched filtering: the matched filter's output peaks at
    the three event locations with amplitudes in the 1 : 0.5 : 0.25
    ratio. This replaces a placeholder that computed a KS statistic on
    the length argument.

    Parameters
    ----------
    n : int, default 36
        Signal length; must cover the last event sample (28).

    Returns
    -------
    RichResult
        keys: ``signal`` (length n), ``pattern`` (g), ``onsets``,
        ``amplitudes``, ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd edn.
    Wiley-IEEE. Ch. 4, Eqs. (4.51)-(4.53), p. 240.
    """
    n = int(n)
    if n < 29:
        raise ValueError(f"n must be at least 29 to hold the third event, got {n}.")
    g = np.array([3.0, 2.0, 1.0])
    onsets = [5, 16, 26]
    amps = [1.0, 0.5, 0.25]
    x = np.zeros(n)
    for o, a in zip(onsets, amps):
        x[o : o + 3] += a * g
    return RichResult(
        payload={
            "signal": x,
            "pattern": g,
            "onsets": onsets,
            "amplitudes": amps,
            "n": n,
            "method": "Rangayyan Ch.4 three-event test signal (Eq. 4.51)",
        }
    )


def cheatsheet():
    return "rng223: Rangayyan Ch.4 three-event matched-filter test signal (Eq. 4.51)"
