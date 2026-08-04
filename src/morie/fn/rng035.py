# morie.fn -- function file (rootcoder007/morie)
"""Discrete-time unit step function (Rangayyan eq. 3.35)."""


from ._richresult import RichResult

__all__ = ["stepseq", "rangayyan_ch3_discrete_unit_step"]


def stepseq(n, shift=0):
    """Discrete-time unit step u(n).

    Rangayyan (2024) eq. (3.35):
        u(n) = 1 for n >= 0, 0 otherwise.

    The inequality is non-strict here, so u(0) = 1 -- the opposite of the
    continuous step of eq. (3.27), where u(0) = 0.  The first difference
    of this sequence is the discrete impulse of eq. (3.34), which is
    returned as a cross-check.
    """
    if isinstance(n, int):
        idx = list(range(n))
    else:
        idx = [int(v) for v in n]
    s = int(shift)
    u = [1.0 if i - s >= 0 else 0.0 for i in idx]
    diff = [u[0]] + [u[i] - u[i - 1] for i in range(1, len(u))]
    return RichResult(payload={
        "u": u, "n": idx, "shift": s, "first_difference": diff,
        "value_at_origin": 1.0,
        "method": "Rangayyan (2024) eq. (3.35)"})


rangayyan_ch3_discrete_unit_step = stepseq  # pre-policy spelling


def cheatsheet():
    return "rng035: discrete unit step, Rangayyan eq. (3.35)"
