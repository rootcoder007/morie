# morie.fn -- function file (rootcoder007/morie)
"""Multi-output DNN for multi-trait genomic prediction.

SOURCE.  Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, doi:10.1007/978-3-030-89010-0.

The ARCHITECTURE is Chapter 12, Section 12.4.1, "DNN with Multivariate
Continuous Outcomes", volume [Pages 477-532], pp.490-493.  That section
gives the model as Keras code rather than as an equation: an input layer,
a stack of shared hidden dense layers each followed by dropout, then ONE
one-unit linear output head per trait, compiled with a per-head loss and a
per-head ``loss_weights`` entry.  That is exactly the model here --
shared hidden stack, T linear heads, joint loss sum_t w_t L_t.

The TRAINING EQUATIONS are Chapter 10, Section 10.8.1, volume
[Pages 379-425], pp.412-413, read from rendered page images because the
text layer of this chapter drops minus signs:

    Step 4   z_ik^(h) = sum_{p=0}^{P} w_kp^(h) x_ip
    Step 5   V_ik^(h) = g^(h)(z_ik^(h))
    Step 6   z_ij^(l) = sum_{k=0}^{M} w_jk^(l) V_ik^(h)
    Step 7   yhat_ij  = g^(l)(z_ij^(l))
    Step 8   E(w) = (1/(2 n L)) sum_i sum_j (yhat_ij - y_ij)^2
    Step 9   delta_ij = (y_ij - yhat_ij) g^(l)-prime(z_ij^(l))
    Step 10  psi_ik   = g^(h)-prime(z_ik^(h)) sum_{j=1}^{L} delta_ij w_jk^(l)
    Step 11  w_jk^(l) <- w_jk^(l) + eta delta_ij V_ik^(h)
    Step 12  w_kp^(h) <- w_kp^(h) + eta psi_ik x_ip

    The sign convention is the one printed: delta carries (y - yhat), not
    (yhat - y), and the updates ADD eta times the gradient term.  The two
    sign flips cancel, so this is ordinary gradient descent on E, but it is
    implemented in the printed form so that the hand computation of Section
    10.8.2 reproduces digit for digit.

    Step 10 also fixes the intercept handling that is easy to get wrong:
    the sum runs over the output weights EXCLUDING the intercept column,
    which p.416 states explicitly -- "where w_1^(l) is w^(l) without the
    weight of the intercept, that is, without the first element".

MULTI-TRAIT LOSS.  Section 12.4.1 weights each head, so the joint loss is
E(w) = (1/(2 n T)) sum_i sum_t w_t (yhat_it - y_it)^2 and the head weight
w_t multiplies delta_it.  At T = 1 with w = 1 this collapses exactly onto
the Chapter 10 loss of Step 8, which is what the anchor exploits.

DEFAULT HEAD WEIGHTS.  p.493 prints a recipe rather than a formula: "(1)
first we calculated the median of each trait, (2) then we calculated the
0.25 and 0.75 quantiles for each trait, (3) then we calculated the maximum
distance in terms of absolute value between the median and both quantiles,
(4) then we used as the weight for the first trait (GY) its calculated
distance, and (5) then we used as weight for the second trait the value
obtained by dividing the distance of the first trait by the distance of
the second trait".  Implemented as printed:

    d_t = max(|median_t - q25_t|, |q75_t - median_t|)
    w_1 = d_1,   w_t = d_1 / d_t  for t >= 2.

    Noted, not silently corrected: step (4) makes w_1 the raw distance
    while every other weight is a ratio, so the overall scale of the loss
    depends on the units of trait 1.  The book itself calls these steps
    "only suggestions that can work for some data sets".  Pass ``heads``
    explicitly to override.

DEPARTURES FROM SECTION 12.4.1, stated rather than papered over:

  * DROPOUT IS NOT IMPLEMENTED.  Dropout is stochastic by definition and
    this package requires both language arms to land on the same numbers.
    The Chapter 12 grid search itself includes dropout1 = 0 as one of its
    two settings, so the dropout-free model is inside the book grid.
  * THE OPTIMISER IS PLAIN FULL-BATCH GRADIENT DESCENT, the Chapter 10
    Steps 11-12 update, not the Adam of the Chapter 12 code.  Adam is
    never given as equations anywhere in the book, only as the Keras call
    ``optimizer_adam(lr=...)``, so implementing it would mean writing from
    memory rather than from the source.
  * There is no validation split and no early stopping on a held-out set;
    training stops on the Chapter 10 Step 14 criterion, E(w) <= tol.

DETERMINISM.  Chapter 10 Step 1 says "initialize the weights to small
random values".  Weights are instead laid down by a linear congruential
generator, seeded by ``seed``, walked in a fixed order over the layers.
An LCG is used rather than the low-discrepancy van der Corput draws
available in ``_s03core`` on purpose: a van der Corput stream strided
across several parameter blocks makes those blocks correlated, which has
already produced a silently wrong module on this shelf, and a correlated
weight initialisation breaks the symmetry-breaking that a hidden layer
needs.  Pass ``init`` to supply weights directly.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["dnn_multitrait"]

_LCG_A = 1103515245
_LCG_C = 12345
_LCG_M = 2147483648


def _act(name, z):
    if name == "linear":
        return z
    if name == "relu":
        return z if z > 0.0 else 0.0
    if name == "sigmoid":
        return core.sigmoid(z)
    if name == "tanh":
        return math.tanh(z)
    raise ValueError("dnn_multitrait: unknown activation %r" % (name,))


def _dact(name, z, g):
    """Derivative of the activation at z, given its value g."""
    if name == "linear":
        return 1.0
    if name == "relu":
        return 1.0 if z > 0.0 else 0.0
    if name == "sigmoid":
        return g * (1.0 - g)
    if name == "tanh":
        return 1.0 - g * g
    raise ValueError("dnn_multitrait: unknown activation %r" % (name,))


def _head_weights(Yc, T, n):
    """The p.493 median/quantile recipe."""
    d = []
    for t in range(T):
        col = [Yc[i][t] for i in range(n)]
        med = core.median(col)
        q1 = core.quantile7(col, 0.25)
        q3 = core.quantile7(col, 0.75)
        dt = max(abs(med - q1), abs(q3 - med))
        if not dt > 0.0:
            raise ValueError(
                "dnn_multitrait: trait %d has a zero interquartile spread, so the "
                "p.493 head-weight recipe divides by zero; pass heads explicitly" % (t + 1)
            )
        d.append(dt)
    return [d[0] if t == 0 else d[0] / d[t] for t in range(T)]


def dnn_multitrait(X, Y, layers, heads=None, activation="relu",
                   out_activation="linear", eta=0.1, epochs=200, tol=0.0,
                   seed=1, init=None):
    """Shared-hidden multi-head DNN trained by the Chapter 10 backpropagation.

    Parameters
    ----------
    X : array-like
        n-by-p matrix of inputs (markers, or any predictors).
    Y : array-like
        n-by-T matrix of trait values; T is the number of output heads.
    layers : sequence of int
        Widths of the shared hidden layers, e.g. ``[33, 33, 33]`` for the
        three-hidden-layer stack of Section 12.4.1.
    heads : sequence of float or None
        Per-head loss weights w_t.  None applies the p.493 recipe.
    activation : {"relu", "sigmoid", "tanh", "linear"}
        Hidden activation; Section 12.4.1 uses relu.
    out_activation : {"linear", "sigmoid", "tanh", "relu"}
        Head activation; Section 12.4.1 uses linear for continuous traits.
    eta : float
        Learning rate of Steps 11-12.
    epochs : int
        Maximum number of full-batch epochs.
    tol : float
        Step 14 stopping tolerance on E(w).  Zero means run all epochs.
    seed : int
        LCG seed for the weight initialisation.
    init : sequence of matrices or None
        Explicit starting weights, one matrix per layer including the
        output layer; layer k has shape (fan_in + 1) by fan_out with the
        intercept in row 0.

    Returns
    -------
    Y_hat : n-by-T matrix of predictions after training
    """
    Xm = core.mat(X)
    n = len(Xm)
    if n == 0:
        raise ValueError("dnn_multitrait: X is empty")
    p = len(Xm[0])
    if p == 0:
        raise ValueError("dnn_multitrait: X has no columns")
    for r in Xm:
        if len(r) != p:
            raise ValueError("dnn_multitrait: rows of X have different lengths")
    Yc = core.mat(Y)
    if len(Yc) != n:
        raise ValueError("dnn_multitrait: X and Y disagree on the number of observations")
    T = len(Yc[0])
    if T == 0:
        raise ValueError("dnn_multitrait: Y has no traits")
    for r in Yc:
        if len(r) != T:
            raise ValueError("dnn_multitrait: rows of Y have different lengths")
    hid = [int(v) for v in layers]
    for v in hid:
        if v < 1:
            raise ValueError("dnn_multitrait: every hidden layer must have at least one unit")
    eta = float(eta)
    if not eta > 0.0:
        raise ValueError("dnn_multitrait: eta must be positive")
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError("dnn_multitrait: epochs must be at least 1")

    if heads is None:
        wt = _head_weights(Yc, T, n)
    else:
        wt = [float(v) for v in core.vec(heads)]
        if len(wt) != T:
            raise ValueError("dnn_multitrait: heads must give one loss weight per column of Y")
        for v in wt:
            if v < 0.0:
                raise ValueError("dnn_multitrait: head loss weights must be non-negative")

    dims = [p] + hid + [T]
    nlay = len(dims) - 1
    if init is not None:
        W = [core.mat(m_) for m_ in init]
        if len(W) != nlay:
            raise ValueError("dnn_multitrait: init must give one weight matrix per layer")
        for k in range(nlay):
            if len(W[k]) != dims[k] + 1 or len(W[k][0]) != dims[k + 1]:
                raise ValueError("dnn_multitrait: init layer %d has the wrong shape" % (k + 1))
    else:
        s = int(seed) % _LCG_M
        W = []
        for k in range(nlay):
            Mk = []
            for _ in range(dims[k] + 1):
                row = []
                for _ in range(dims[k + 1]):
                    s = (_LCG_A * s + _LCG_C) % _LCG_M
                    row.append(0.2 * (s / float(_LCG_M)) - 0.1)
                Mk.append(row)
            W.append(Mk)

    acts = [activation] * len(hid) + [out_activation]
    loss = float("nan")
    Yhat = [[0.0] * T for _ in range(n)]
    ran = 0
    for _ep in range(epochs):
        # Steps 4-7, forward
        Z = []
        A = [[[1.0] + list(Xm[i]) for i in range(n)]]
        for k in range(nlay):
            zk = [[sum(A[k][i][q] * W[k][q][j] for q in range(dims[k] + 1))
                   for j in range(dims[k + 1])] for i in range(n)]
            gk = [[_act(acts[k], zk[i][j]) for j in range(dims[k + 1])] for i in range(n)]
            Z.append(zk)
            if k < nlay - 1:
                A.append([[1.0] + gk[i] for i in range(n)])
            else:
                Yhat = gk
        # Step 8, evaluated at the CURRENT weights, as in the book
        acc = 0.0
        for i in range(n):
            for t in range(T):
                d = Yhat[i][t] - Yc[i][t]
                acc += wt[t] * d * d
        loss = acc / (2.0 * n * T)
        ran = _ep + 1
        if tol > 0.0 and loss <= tol:
            break
        # Step 9, output deltas
        D = [[wt[t] * (Yc[i][t] - Yhat[i][t]) * _dact(acts[nlay - 1], Z[nlay - 1][i][t], Yhat[i][t])
              for t in range(T)] for i in range(n)]
        # Steps 10-12: all deltas from the OLD weights, then update
        newW = [None] * nlay
        Dk = D
        for k in range(nlay - 1, -1, -1):
            newW[k] = [[W[k][q][j] + eta * sum(A[k][i][q] * Dk[i][j] for i in range(n))
                        for j in range(dims[k + 1])] for q in range(dims[k] + 1)]
            if k > 0:
                # Step 10; row 0 of W[k] is the intercept and is excluded
                Dk = [[_dact(acts[k - 1], Z[k - 1][i][q], A[k][i][q + 1])
                       * sum(Dk[i][j] * W[k][q + 1][j] for j in range(dims[k + 1]))
                       for q in range(dims[k])] for i in range(n)]
        W = newW

    return RichResult(
        title="Multi-trait DNN",
        summary_lines=[("obs", n), ("traits", T), ("hidden", hid), ("epochs", ran)],
        payload={
            "estimate": loss,
            "Y_hat": Yhat,
            "loss": loss,
            "head_weights": wt,
            "weights": W,
            "epochs_run": ran,
            "n": n,
            "method": (
                "shared hidden stack with one linear head per trait (Ch 12 Sec 12.4.1, "
                "pp.490-493), trained by the Ch 10 Sec 10.8.1 backpropagation Steps 4-12 "
                "(pp.412-413); joint loss sum_t w_t L_t; Montesinos Lopez et al. (2022)"
            ),
        },
    )


def cheatsheet():
    return "dnnmt: shared-hidden multi-head DNN, Ch 12 Sec 12.4.1 trained by the Ch 10 backprop"


# compact alias per ledger/NAMING.md
dnnmultitrait = dnn_multitrait
