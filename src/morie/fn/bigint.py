# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact integer arithmetic, and the combinatorial counts that need it.

Python integers are already arbitrary precision, so this module is thin
on the Python side. It exists because **R is not**, and the R
counterpart in R/bigint_native.R implements base-10^6 limb arithmetic
from scratch to match what is here.

The reason is not tidiness. In R:

    2^53 + 1 == 2^53                             is TRUE
    choose(100, 50) -> 100891344545563076171808112640
    the exact value    100891344545564193334812497256

The two part company at the thirteenth significant digit and nothing in
the output says so. Every exact count in the combinatorics shelves runs
through this pairing instead, and the parity tests compare **decimal
strings**, not doubles, so a silent loss of low-order digits fails the
test rather than passing it.

gmp is deliberately not used on either side. These are elementary
operations and the package is a native specialization.
"""

import math

from ._richresult import RichResult

__all__ = [
    "big_factorial",
    "big_binomial",
    "big_pow",
    "big_digits",
    "fits_double",
    "below_double_threshold",
    "exact_or_flag",
]

_DOUBLE_EXACT = 2 ** 53


def big_factorial(n):
    """Exact ``n!`` as an integer.

    Examples
    --------
    >>> big_factorial(20)
    2432902008176640000
    >>> big_digits(big_factorial(100))
    158
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be a non-negative whole number; got {n}.")
    return math.factorial(n)


def big_binomial(n, k):
    r"""Exact :math:`\binom{n}{k}`.

    Examples
    --------
    >>> big_binomial(100, 50)
    100891344545564193334812497256
    >>> big_binomial(5, 7)
    0
    """
    n, k = int(n), int(k)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def big_pow(a, k):
    """Exact ``a ** k``."""
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be a non-negative whole number; got {k}.")
    return int(a) ** k


def big_digits(a):
    """Number of decimal digits, ignoring any sign."""
    return len(str(abs(int(a))))


def fits_double(a):
    """Is this value exactly representable as an IEEE double?

    Tested by round-tripping, not by comparing against
    :math:`2^{53}`. The threshold rule is sufficient but **not
    necessary**, and treating it as necessary mislabels exact values as
    lossy: :math:`20! = 2432902008176640000` is about
    :math:`2.4 \times 10^{18}`, far above :math:`2^{53}`, and is still
    exactly representable because it is :math:`2^{18}` times an odd
    number, so its low-order bits are already zero. Below
    :math:`2^{53}` every integer round-trips; above it some do and most
    do not, and only the round trip knows which.

    Examples
    --------
    >>> fits_double(2 ** 53)
    True
    >>> fits_double(2 ** 53 + 1)
    False
    >>> fits_double(big_factorial(20))          # above 2^53, still exact
    True
    >>> fits_double(big_binomial(100, 50))
    False
    """
    v = int(a)
    try:
        return int(float(v)) == v
    except OverflowError:
        return False


def below_double_threshold(a):
    """Is ``|a|`` at or below :math:`2^{53}`?

    The conservative sufficient condition. Every integer this size
    round-trips through a double; see :func:`fits_double` for the exact
    test.
    """
    return abs(int(a)) <= _DOUBLE_EXACT


def exact_or_flag(value, label="value"):
    """Wrap an exact integer with an explicit precision-loss flag.

    Returns the value both as an exact decimal string and as a double,
    together with ``exact_as_double`` and, when that is False, the
    absolute error the double conversion introduces. Reporting the
    error rather than the flag alone matters: for
    :math:`\\binom{100}{50}` the double is wrong by more than
    :math:`10^{15}`, which is not a rounding artefact anyone should
    absorb silently.

    Returns
    -------
    RichResult with ``exact`` (str), ``as_double``, ``exact_as_double``,
    ``absolute_error``, ``n_digits``.
    """
    v = int(value)
    ok = fits_double(v)
    try:
        dbl = float(v)
        err = 0 if ok else abs(v - int(dbl))
    except OverflowError:
        dbl = float("inf")
        err = v
    out = RichResult(
        title=f"Exact {label}",
        summary_lines=[
            ("Exact", str(v)),
            ("Digits", big_digits(v)),
            ("Exact as a double", ok),
        ],
        payload={
            "exact": str(v),
            "value": v,
            "estimate": dbl,
            "as_double": dbl,
            "exact_as_double": ok,
            "absolute_error": err,
            "n_digits": big_digits(v),
            "n": big_digits(v),
            "method": "Exact integer with a precision-loss flag",
        },
    )
    if not ok:
        out.warnings.append(
            f"This {label} has {big_digits(v)} digits and exceeds 2^53, so "
            f"the double conversion is wrong by {err}. Use the exact string; "
            "R's own choose() makes precisely this error and reports nothing."
        )
    return out


def cheatsheet():
    return (
        "bigint: exact integer arithmetic and a precision-loss flag, paired "
        "with a from-scratch base-10^6 implementation on the R side"
    )
