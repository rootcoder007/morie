# morie.fn -- function file (rootcoder007/morie)
"""Discrete-time unit impulse function (Rangayyan eq. 3.34)."""


from ._richresult import RichResult

__all__ = ["kdelta", "rangayyan_ch3_discrete_delta"]


def kdelta(n, shift=0, amplitude=1.0):
    """Discrete-time unit impulse delta(n).

    Rangayyan (2024) eq. (3.34):
        delta(n) = 1 if n = 0, 0 otherwise.

    Unlike the continuous delta of eq. (3.24) this is an ordinary
    sequence with a finite value at the origin, so it can be evaluated
    rather than only approximated.  Figure 3.13 shows the shifted and
    scaled versions ``shift`` and ``amplitude`` produce.

    Parameters
    ----------
    n : int or sequence of int
        Sample indices to evaluate at; an integer N is read as the range
        0, 1, ..., N-1.
    shift : int
        Location of the impulse (the n0 of delta(n - n0)).
    amplitude : float
        Scale factor.
    """
    if isinstance(n, int):
        idx = list(range(n))
    else:
        idx = [int(v) for v in n]
    s, a = int(shift), float(amplitude)
    return RichResult(payload={
        "delta": [a if i == s else 0.0 for i in idx], "n": idx,
        "shift": s, "amplitude": a,
        "method": "Rangayyan (2024) eq. (3.34)"})


rangayyan_ch3_discrete_delta = kdelta  # pre-policy spelling


def cheatsheet():
    return "rng034: discrete unit impulse, Rangayyan eq. (3.34)"
