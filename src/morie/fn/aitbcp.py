# morie.fn -- function file (rootcoder007/morie)
"""Bray-Curtis dissimilarity between two closed compositions.

Source: Bray, J. R. and Curtis, J. T. (1957), "An ordination of the
upland forest communities of southern Wisconsin", *Ecological
Monographs* 27(4):325-349.  The coefficient is

    BC(x, y) = sum_i |x_i - y_i| / sum_i (x_i + y_i)

taking values in [0, 1] for non-negative x and y: 0 when the two
vectors are identical, 1 when they share no part.

"Closed" in the module name is the compositional-data sense of
Aitchison: each vector is divided by its own total before the
coefficient is formed, so that only the RELATIVE composition matters and
a change of sampling effort cannot move the answer.  With ``close=True``
(the default) the denominator is therefore exactly 2 and BC reduces to
half the L1 distance between the two closed vectors.  Pass
``close=False`` for the raw abundance form of the 1957 paper, in which
total abundance does contribute.

Bray-Curtis is a dissimilarity, not a distance: it does not satisfy the
triangle inequality, so it must not be fed to a routine that assumes a
metric.  This module states that rather than silently permitting it.
"""

from ._richresult import RichResult

__all__ = ["compositional_bray_curtis"]


def compositional_bray_curtis(x, y, close=True):
    """Bray-Curtis dissimilarity between two non-negative vectors.

    Parameters
    ----------
    x, y : array-like
        Non-negative parts of equal length.
    close : bool
        Divide each vector by its own total first (the compositional
        convention).  ``False`` gives the raw abundance coefficient.

    Returns
    -------
    RichResult
        ``bc``, ``estimate`` (the same number), ``numerator``,
        ``denominator``, ``similarity`` (= 1 - bc), ``n``.
    """
    a = [float(v) for v in x]
    b = [float(v) for v in y]
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("x and y must be non-empty and of equal length")
    for v in a + b:
        if v < 0.0:
            raise ValueError("Bray-Curtis is defined for non-negative parts")
    if close:
        sa = sum(a)
        sb = sum(b)
        if sa <= 0.0 or sb <= 0.0:
            raise ValueError("a composition cannot be closed to a zero total")
        a = [v / sa for v in a]
        b = [v / sb for v in b]
    num = 0.0
    den = 0.0
    for i in range(n):
        dv = a[i] - b[i]
        num += dv if dv >= 0.0 else -dv
        den += a[i] + b[i]
    if den <= 0.0:
        raise ValueError("both vectors are zero; Bray-Curtis is undefined")
    bc = num / den
    return RichResult(payload={
        "bc": float(bc), "estimate": float(bc), "numerator": float(num),
        "denominator": float(den), "similarity": float(1.0 - bc),
        "closed": bool(close), "n": n,
        "method": "Bray & Curtis (1957) dissimilarity, "
                  "sum|x-y| / sum(x+y); a dissimilarity, not a metric"})


def cheatsheet():
    return "aitbcp: Bray & Curtis (1957) dissimilarity on closed compositions"
