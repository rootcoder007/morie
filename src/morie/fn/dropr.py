# morie.fn -- function file (rootcoder007/morie)
"""Apply a dropout mask with inverted scaling."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['dropmask', 'dropout_regularization']


def dropmask(x, mask, rate):
    """Apply a dropout mask with inverted scaling.

    Formula: a_i = x_i * m_i / (1 - rate),  m_i in {0, 1} supplied by the caller

    Parameters
    ----------
    x : array-like
        Activations of the layer being regularized.
    mask : array-like
        Keep/drop indicator per unit: 1 keeps, 0 drops.  Supplied by the caller so the result is reproducible.
    rate : float
        Dropout rate in [0, 1); the surviving activations are divided by 1 - rate.

    Returns
    -------
    RichResult
        ``activation``, ``kept``, ``dropped``, ``rate``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.6 p. 404 describes dropout as setting a random fraction of the weights of the input or hidden neurons to zero, so their contribution is removed on the forward pass and they receive no update on the backward pass; it prints no formula and attributes the method to Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014), Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 15:1929-1958, which is where the 1/(1 - rate) inverted scaling comes from.  The mask is an argument rather than drawn internally so the function is deterministic.  Chapter read from the PDF; the scaling is from the paper the book names.
    """
    x = C.vec(x)
    m = C.vec(mask)
    rate = float(rate)
    if len(x) != len(m):
        raise ValueError("x and mask must have the same length")
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must lie in [0, 1)")
    if any(v not in (0.0, 1.0) for v in m):
        raise ValueError("mask entries must be 0 or 1")
    s = 1.0 / (1.0 - rate)
    kept = int(sum(m))
    return RichResult(payload={
        "activation": [a * b * s for a, b in zip(x, m)],
        "kept": kept, "dropped": len(x) - kept, "rate": rate, "n": len(x),
        "method": "Inverted dropout, MVSML Sect. 10.6"})


dropout_regularization = dropmask


def cheatsheet():
    return 'dropr: Apply a dropout mask with inverted scaling.'
