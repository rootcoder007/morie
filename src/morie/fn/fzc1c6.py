# morie.fn -- function file (rootcoder007/morie)
"""Conditions C1-C6 for boundary-free MRL estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_conditions_c1_c6"]


def fauzi_conditions_c1_c6(x=None, transform="log", check_moments=True):
    r"""Conditions C1-C6 of Fauzi Ch. 4, the assumptions under which
    every theorem in the chapter holds.

    From the printed text:

    * **C1-C2** -- standard kernel conditions.
    * **C3** -- needed for the BIJECTIVITY and simplicity of the
      transformation. Without it :math:`g^{-1}` is not well defined
      and the whole construction collapses.
    * **C4** -- required for the validity of the serial expansions
      the proofs use.
    * **C5** -- the integrals :math:`\int g'(ux)K(x)dx` and
      :math:`\int g'(ux)V(x)dx` are finite for all :math:`u` in a
      neighbourhood of the origin.
    * **C6** -- the moments :math:`E(X)`, :math:`E(X^2)` and
      :math:`E(X^3)` exist.

    C5 and C6 are the ones that bite in practice, and the book says
    why: they are what make the bias and variance formulas derivable
    at all. C6 in particular rules out heavy-tailed data -- a
    distribution without a third moment is outside the theory, however
    well the estimator appears to behave on a sample from it. The
    module checks the empirical moments and flags that risk rather
    than leaving the assumption implicit.

    Parameters
    ----------
    x : array-like, optional
        Sample, for the empirical moment check.
    transform : {"log", "identity"}
        The bijection, checked for C3.
    check_moments : bool
        Run the C6 diagnostic.

    Returns
    -------
    RichResult
        keys: ``conditions``, ``C3_bijective``, ``C6_moments``,
        ``heavy_tail_warning``, ``binding_in_practice``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), conditions C1-C6, Ch. 4. From the PDF.
    """
    from ._fauzi import boundary_free_transform

    tr = boundary_free_transform(transform)
    conds = {
        "C1": "standard kernel condition",
        "C2": "standard kernel condition",
        "C3": "bijectivity and simplicity of the transformation g",
        "C4": "validity of the serial expansions used in the proofs",
        "C5": "int g'(ux)K(x)dx and int g'(ux)V(x)dx finite near the origin",
        "C6": "E(X), E(X^2) and E(X^3) exist",
    }
    moments = None
    heavy = None
    if x is not None and check_moments:
        xv = np.asarray(x, dtype=float).ravel()
        if xv.size < 4:
            raise ValueError(f"need at least 4 observations, got {xv.size}.")
        moments = {"E_X": float(np.mean(xv)),
                   "E_X2": float(np.mean(xv ** 2)),
                   "E_X3": float(np.mean(xv ** 3))}
        # a crude tail check: compare the empirical third moment with
        # what a same-variance normal would give
        sd = float(np.std(xv, ddof=1))
        m = moments["E_X"]
        ref = m ** 3 + 3 * m * sd ** 2
        heavy = bool(ref > 0 and moments["E_X3"] > 20 * ref)
    return RichResult(payload={
        "conditions": conds, "C3_bijective": True,
        "C6_moments": moments, "heavy_tail_warning": heavy,
        "binding_in_practice": ["C5", "C6"],
        "why": "C5 and C6 are what make the bias and variance formulas "
               "derivable; C6 rules out heavy-tailed data entirely",
        "transform": tr["name"],
        "method": "Conditions C1-C6 of Ch. 4, with the C6 moment check made explicit"})


def cheatsheet():
    return "fzc1c6: C6 (three moments) rules out heavy tails -- the theory simply does not cover them"
