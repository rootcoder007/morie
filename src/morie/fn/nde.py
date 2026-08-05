# morie.fn -- function file (rootcoder007/morie)
"""Natural direct effect from a linear structural mediation model."""

from . import _tail1core as C
from .tmlnde import ndeff

from ._richresult import RichResult

__all__ = ["natural_direct_effect", "naturaldirecteffect"]


def natural_direct_effect(X, M, Y):
    """Pearl's natural direct effect, estimated from a linear SEM.

        NDE = E[Y(1, M(0))] - E[Y(0, M(0))]

    The contrast itself is not recomputed here: it is delegated to
    ``tmlnde.ndeff``, which is the single implementation of the NDE
    contrast in this package.  What this module adds is the step
    ``tmlnde`` deliberately refuses to take -- producing the two
    cross-world quantities from data, which requires a model.

    The model is the standard two-equation linear SEM with an
    exposure-mediator interaction (VanderWeele 2015, eq. 2.9-2.11):

        M = b0 + b1 X + e_M
        Y = c0 + c1 X + c2 M + c3 X M + e_Y

    Under sequential ignorability, ``E[M(0)] = b0``, so

        E[Y(x, M(0))] = c0 + c1 x + c2 b0 + c3 x b0
        NDE = c1 + c3 b0,      NIE = b1 (c2 + c3),

    contrasting x = 1 against x = 0.  ``NDE + NIE`` is the total effect
    exactly; that identity is what the decomposition exists for and is
    reported as ``total``.  With no interaction (``c3 = 0``) the NDE
    collapses to the coefficient on X in the regression of Y on X and M,
    which is the classical Baron-Kenny direct path.

    No standard error is reported.  ``se`` is NA because the two
    cross-world means are point predictions from the fitted SEM, not a
    sample of contrasts; reporting the spread of a constant vector as a
    standard error would be a lie about what is known.

    Parameters
    ----------
    X : array-like
        Exposure.
    M : array-like
        Mediator.
    Y : array-like
        Outcome.

    Returns
    -------
    RichResult
        ``estimate`` (NDE), ``se``, ``nde``, ``nie``, ``total``,
        ``mean_y10``, ``mean_y00``, ``b0``, ``b1``, ``c0``, ``c1``,
        ``c2``, ``c3``, ``n``.

    References
    ----------
    Pearl, J. (2001), "Direct and indirect effects", Proceedings of the
    17th Conference on Uncertainty in Artificial Intelligence, 411-420.
    VanderWeele, T. J. (2015), Explanation in Causal Inference: Methods
    for Mediation and Interaction, Oxford University Press.  Standard
    published form of the linear-SEM decomposition; neither source was in
    the local corpus, so the equations are stated in full above.
    """
    x = C.vec(X)
    m = C.vec(M)
    y = C.vec(Y)
    n = len(x)
    if n == 0:
        raise ValueError("X is empty")
    if len(m) != n or len(y) != n:
        raise ValueError("X, M and Y must have the same length")
    if n < 4:
        raise ValueError("need at least 4 observations to fit the SEM")
    dm = [[1.0, x[i]] for i in range(n)]
    b = C.lstsq(dm, m)[0]
    b0, b1 = float(b[0]), float(b[1])
    dy = [[1.0, x[i], m[i], x[i] * m[i]] for i in range(n)]
    c = C.lstsq(dy, y)[0]
    c0, c1, c2, c3 = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    y10 = c0 + c1 + c2 * b0 + c3 * b0
    y00 = c0 + c2 * b0
    r = ndeff([y10], [y00])
    nie = b1 * (c2 + c3)
    return RichResult(payload={
        "estimate": r["estimate"], "se": r["se"], "nde": r["estimate"],
        "nie": nie, "total": r["estimate"] + nie,
        "mean_y10": r["mean_y10"], "mean_y00": r["mean_y00"],
        "b0": b0, "b1": b1, "c0": c0, "c1": c1, "c2": c2, "c3": c3,
        "n": n, "method": "Natural direct effect (linear SEM, Pearl 2001)"})


naturaldirecteffect = natural_direct_effect


def cheatsheet():
    return "nde: Natural direct effect from a linear structural mediation model"
