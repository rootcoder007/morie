# morie.fn -- function file (rootcoder007/morie)
"""Decomposition of a signal into weighted deltas (Rangayyan eq. 3.29)."""


from ._rgcore import aslist, gridint
from ._richresult import RichResult

__all__ = ["deltadecomp", "rangayyan_ch3_signal_as_delta_decomposition"]


def deltadecomp(x, t=None):
    """Resolve a signal into a weighted combination of shifted deltas.

    Rangayyan (2024) eq. (3.29):
        x(t) = integral x(alpha) delta(t - alpha) d alpha.

    The book reads this as resolving x into mutually orthogonal delta
    functions.  Discretely, the weight carried by the delta at alpha_i is
    x(alpha_i) times the grid spacing, so that summing the weights
    reproduces the integral of x rather than the sum of its samples;
    reconstructing from the weights returns the original samples exactly,
    which is the check returned in ``reconstruction_error``.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    ts = [float(i) for i in range(n)] if t is None else aslist(t)
    if len(ts) != n:
        raise ValueError("t and x must have the same length")
    if n == 1:
        dt = [1.0]
    else:
        # trapezoidal weights: half a spacing at each end, so that the
        # weights sum to the integral of x rather than overcounting the
        # two endpoints by half a panel each.
        dt = []
        for i in range(n):
            lo = ts[i] - ts[i - 1] if i > 0 else 0.0
            hi = ts[i + 1] - ts[i] if i < n - 1 else 0.0
            dt.append(0.5 * (lo + hi))
    weights = [v * d for v, d in zip(xs, dt)]
    recon = [w / d for w, d in zip(weights, dt)]
    err = max(abs(a - b) for a, b in zip(recon, xs))
    return RichResult(payload={
        "locations": ts, "weights": weights, "amplitudes": xs,
        "total_weight": sum(weights),
        "integral": gridint(xs, ts) if n > 1 else 0.0,
        "reconstruction_error": err,
        "method": "Rangayyan (2024) eq. (3.29)"})


rangayyan_ch3_signal_as_delta_decomposition = deltadecomp  # pre-policy spelling


def cheatsheet():
    return "rng029: delta decomposition of a signal, Rangayyan eq. (3.29)"
