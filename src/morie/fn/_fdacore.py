# morie.fn -- function file (rootcoder007/morie)
"""Private numeric helpers shared by the functional-data function files.

The mirror of R/aaa_helpers_fda.R.  Every routine performs the same
floating-point operations in the same order as its R counterpart, which
is what lets the parity harness assert agreement at 1e-9.

The integration rule here always runs over the WHOLE grid.  A sibling
module in this package once integrated over [a+h, b-h], dropping both
end intervals, and returned 3.8667 where the closed form is 4; both
arms had the same defect, so parity was green and only a closed-form
anchor caught it.  Nothing in this file may narrow the interval.
"""

from __future__ import annotations

__all__: list[str] = []


def trapz(t, v):
    """Composite trapezoid rule for v sampled at t, over the whole of t."""
    s = 0.0
    for i in range(len(t) - 1):
        s += 0.5 * (v[i] + v[i + 1]) * (t[i + 1] - t[i])
    return s


def grid(n):
    """The default equally spaced grid on [0, 1] with n points."""
    return [i / float(n - 1) for i in range(n)]


def colmeans(A, nr, nc):
    """Column means of a list-of-rows matrix, summed in row order."""
    m = [0.0] * nc
    for i in range(nr):
        for j in range(nc):
            m[j] += A[i][j]
    return [v / float(nr) for v in m]
