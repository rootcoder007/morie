# morie.fn -- slice s04 (rootcoder007/morie)
"""Perceptron activation function (step function).

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 379-425], Chapter 10, Section
10.2, "The Building Blocks of Artificial Neural Networks", p. 382-383.
The chapter defines the net input of a neuron as

    v_j = sum_j omega_ij x_j

and its output as y_j = g(v_j), where g is the activation function; "if
we define this function as a unit step (also called threshold), the
output will be 1 if the net input is greater than zero; otherwise the
output will be 0".  Note the strict inequality: the book puts v = 0 in
the 0 branch, and this implementation does the same.

The bias b enters the net input as the weight on the constant input, the
way Section 10.8 writes z = sum_{p=0}^P w_kp x_ip with x_i0 = 1.

The misclassification update w <- w + eta*y_i*x_i named in the function
docstring is Rosenblatt's, not the book's -- Section 10.2 does not give
a learning rule for the single unit; it goes straight to backpropagation
in Section 10.8.  The update is therefore reported as an increment the
caller may apply, computed from the book's net input, and it is labelled
as such in the payload.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["perceptron_activation"]


def perceptron_activation(X, w, b, y=None, eta=1.0):
    """Net input and unit-step activation of a perceptron layer.

    Parameters
    ----------
    X : array-like
        n-by-p matrix of inputs, one pattern per row.
    w : array-like
        Weight vector of length p.
    b : float
        Bias, the weight on the constant input.
    y : array-like, optional
        Targets coded -1/+1.  When given, the Rosenblatt increment
        eta * y_i * x_i is accumulated over the misclassified patterns.
    eta : float
        Learning rate for that increment.

    Returns
    -------
    estimate : the activation of the first pattern
    a        : the unit-step activations, 1 if v > 0 else 0
    v        : the net inputs
    sign     : sign(v), the -1/0/+1 coding
    update   : the accumulated weight increment (zeros when y is absent)
    """
    XX = k.mat(X)
    ww = k.vec(w)
    bb = k.vec(b)
    if not XX:
        raise ValueError("perceptron_activation: X is empty")
    p = len(XX[0])
    if len(ww) != p:
        raise ValueError("perceptron_activation: w does not match the columns of X")
    if len(bb) != 1:
        raise ValueError("perceptron_activation: b must be a single value")
    b0 = bb[0]
    yy = k.vec(y) if y is not None else None
    if yy is not None and len(yy) != len(XX):
        raise ValueError("perceptron_activation: y does not match the rows of X")
    v = []
    a = []
    sg = []
    upd = [0.0] * p
    upd_b = 0.0
    for i in range(len(XX)):
        s = b0
        for j in range(p):
            s += XX[i][j] * ww[j]
        v.append(s)
        a.append(1.0 if s > 0.0 else 0.0)
        sg.append(1.0 if s > 0.0 else (-1.0 if s < 0.0 else 0.0))
        if yy is not None and yy[i] * s <= 0.0:
            for j in range(p):
                upd[j] += float(eta) * yy[i] * XX[i][j]
            upd_b += float(eta) * yy[i]
    return RichResult(
        title="Perceptron activation",
        summary_lines=[("patterns", len(XX)), ("inputs", p)],
        payload={
            "estimate": a[0],
            "a": a,
            "v": v,
            "sign": sg,
            "update": upd,
            "update_b": upd_b,
            "n": len(XX),
            "method": "v = Xw + b with the unit-step g of Chapter 10 Sect. 10.2 (1 if v > 0, else 0)",
        },
    )


def cheatsheet():
    return "percn: Perceptron activation function (step function)"
