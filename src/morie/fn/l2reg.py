# morie.fn -- function file (rootcoder007/morie)
"""Ridge (weight-decay) regularized loss."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['l2pen', 'l2_weight_regularization']


def l2pen(loss, w, lam):
    """Ridge (weight-decay) regularized loss.

    Formula: L(w, lambda) = L(w) + 0.5 * lambda * w'w

    Parameters
    ----------
    loss : float
        Unregularized loss L(w).
    w : array-like
        Network weights.
    lam : float
        Penalty strength lambda; must be non-negative.

    Returns
    -------
    RichResult
        ``penalized_loss``, ``penalty``, ``ep``, ``lambda``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.7.3 p. 403: L(w, lambda) = L(w) + 0.5 * lambda * E_P, with E_P = w'w for the ridge (weight decay, L2) penalty.  Read from the chapter PDF, not recalled.
    """
    w = C.vec(w)
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lambda must be non-negative")
    ep = sum(v * v for v in w)
    pen = 0.5 * lam * ep
    return RichResult(payload={
        "penalized_loss": float(loss) + pen, "penalty": pen, "ep": ep,
        "lambda": lam, "p": len(w),
        "method": "L2 (ridge) regularized loss, MVSML Sect. 10.7.3"})


l2_weight_regularization = l2pen


def cheatsheet():
    return 'l2reg: Ridge (weight-decay) regularized loss.'
