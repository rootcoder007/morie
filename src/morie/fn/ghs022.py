# morie.fn -- slice k04 (rootcoder007/morie)
"""Cell counts N_epsilon for a tail-free / Polya-tree partition.

Source READ FROM THE CORPUS PDF: Ghosal, S. and van der Vaart, A.
(2017), *Fundamentals of Nonparametric Bayesian Inference*, chapter 3
(tail-free and Polya tree priors).  The quantity is the sufficient
statistic that carries all the data's information about a tail-free
prior: for a measurable set A_epsilon in the partition at level
epsilon,

    N_epsilon := # { 1 <= i <= n : X_i in A_epsilon }

and, because a tail-free prior's partition-level masses are
independent, the vector of cell counts at each level is what the
posterior update actually consumes.  Nothing is estimated here; this
is a count.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  That was
particularly bad here: a KS statistic against a fitted normal has no
relationship whatsoever to a partition cell count, and it returned a
plausible number in [0, 1] where an integer count was expected.

The previous KS body is deleted.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_cell_counts"]


def _membership(x, A):
    """True where x falls in the cell A.

    ``A`` is either a callable predicate, or a ``(lo, hi)`` pair read as
    the half-open interval ``[lo, hi)`` -- half-open so that a partition
    given as consecutive breakpoints tiles the line without double
    counting.
    """
    if callable(A):
        return [bool(A(float(v))) for v in x]
    lo, hi = A
    lo = float(lo)
    hi = float(hi)
    return [bool(lo <= float(v) < hi) for v in x]


def ghosal_ch3_tailfree_cell_counts(X_i, A_epsilon, n=None):
    """Count how many observations fall in each partition cell.

    Parameters
    ----------
    X_i : array-like
        The sample.
    A_epsilon : cell, or sequence of cells
        A cell is a callable predicate or a ``(lo, hi)`` half-open
        interval.  Pass a sequence to count a whole partition level at
        once.
    n : int, optional
        Use only the first ``n`` observations, as in the
        ``1 <= i <= n`` of the definition.  Defaults to all of them.

    Returns
    -------
    RichResult
        keys: ``N_epsilon`` (int, or an array of ints when a sequence of
        cells was passed), ``proportion``, ``n``, ``method``.
    """
    x = np.asarray(X_i, dtype=float).ravel()
    if n is not None:
        nn = int(n)
        if nn < 0 or nn > x.size:
            raise ValueError("n must lie in 0..len(X_i)")
        x = x[:nn]
    nn = int(x.size)

    cells = A_epsilon
    single = callable(cells) or (
        isinstance(cells, tuple) and len(cells) == 2 and not isinstance(cells[0], (list, tuple))
    )
    if single:
        cells = [A_epsilon]

    counts = [int(sum(_membership(x, A))) for A in cells]
    if single:
        N = counts[0]
        prop = (N / nn) if nn else float("nan")
    else:
        N = np.array(counts, dtype=int)
        prop = np.array([c / nn for c in counts], dtype=float) if nn else np.array(
            [float("nan")] * len(counts), dtype=float
        )
    return RichResult(
        payload={
            "N_epsilon": N,
            "proportion": prop,
            "n": nn,
            "method": "Tail-free partition cell counts (Ghosal and van der Vaart 2017, ch. 3)",
        }
    )


def cheatsheet():
    return "ghs022: tail-free partition cell counts N_epsilon"
