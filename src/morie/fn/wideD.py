"""Wide & Deep jointly trained classifier."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wide_and_deep"]

_LCG_M = 2147483647.0
_LCG_A = 16807.0


def _lcg(state):
    """One MINSTD step.  Exact in double precision (a * m < 2 ** 53), so
    both language arms produce bit-identical initial weights."""
    return (_LCG_A * state) % _LCG_M


def wide_and_deep(X_wide, X_deep, y, hidden=(8,), epochs=300, lr=0.05,
                  seed=1, crosses=None, l2=0.0):
    """
    Wide & Deep

    Formula: P(Y=1|x) = sigmoid(w_wide' [x, phi(x)] + w_deep' a^(lf) + b)

    The wide part is a generalised linear model on the raw wide features
    and their cross-product transformations, and the deep part is a
    feed-forward network on the dense features.  Their output log-odds
    are summed and fed to one logistic loss, so the two parts are trained
    jointly rather than ensembled: every parameter, in both parts and in
    the shared bias, sees the same gradient step.

    In the numbering of Cheng et al. (2016):

    * eq. (1) -- the cross-product transformation
      ``phi_k(x) = prod_i x_i ** c_ki``, which is what lets the linear
      wide part express feature interactions.
    * eq. (2) -- each hidden layer computes
      ``a^(l+1) = f(W^(l) a^(l) + b^(l))`` with ``f`` the rectifier.
    * eq. (3) -- the prediction above, combining the two parts.

    The paper trains with mini-batch stochastic gradient descent, FTRL
    with L1 for the wide part and AdaGrad for the deep part.  This
    implementation instead uses full-batch gradient descent with a fixed
    step size for a fixed number of epochs, and initialises from a
    reproducible linear congruential generator rather than a random one.
    The model, eq. (1) to (3), and the joint logistic objective are the
    paper's; only the optimiser is not, and it is replaced so that the
    fit is a deterministic function of the inputs.  Do not read the
    result as reproducing the paper's benchmark numbers, which depend on
    that optimiser and on embeddings this function does not fit.

    Parameters
    ----------
    X_wide : array-like
        ``(n, p_wide)`` wide features.
    X_deep : array-like
        ``(n, p_deep)`` dense features for the network.
    y : array-like
        Binary labels in {0, 1}, length ``n``.
    hidden : sequence of int
        Widths of the hidden layers.  Default one layer of 8 units.
    epochs : int
        Number of full-batch gradient steps.
    lr : float
        Step size.
    seed : int
        Seed of the deterministic generator used to initialise weights.
    crosses : sequence of (int, int), optional
        Index pairs into the columns of ``X_wide``.  Each pair adds the
        eq. (1) product of those two columns to the wide design.
    l2 : float
        Ridge penalty on every weight (not on biases).  Default 0.

    Returns
    -------
    result : RichResult
        Keys: coef_wide, coef_deep, bias, hidden_weights, hidden_bias,
        fitted, loss, n, n_wide, n_deep, epochs, method.

    References
    ----------
    Cheng H-T, Koc L, Harmsen J, Shaked T, Chandra T, Aradhye H,
    Anderson G, Corrado G, Chai W, Ispir M, Anil R, Haque Z, Hong L,
    Jain V, Liu X & Shah H (2016).  Wide & Deep learning for recommender
    systems.  Proceedings of the 1st Workshop on Deep Learning for
    Recommender Systems (DLRS 2016), 7-10.  arXiv:1606.07792.
    Equations (1), (2) and (3).
    """
    def _mat(a, name):
        arr = np.asarray(a, dtype=float).tolist()
        if len(arr) == 0:
            raise ValueError(name + " must be non-empty")
        if not isinstance(arr[0], list):
            return [[float(v)] for v in arr]
        return [[float(v) for v in row] for row in arr]

    xw = _mat(X_wide, "X_wide")
    xd = _mat(X_deep, "X_deep")
    yv = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float)).tolist()]
    n = len(yv)
    if len(xw) != n or len(xd) != n:
        raise ValueError("X_wide, X_deep and y must have the same number of rows")
    if n < 2:
        raise ValueError("need at least two observations")
    for v in yv:
        if v != 0.0 and v != 1.0:
            raise ValueError("y must be binary, coded 0 and 1")

    # eq. (1): append the requested cross-product transformations.
    if crosses:
        pw0 = len(xw[0])
        pairs = [(int(a), int(b)) for a, b in crosses]
        for a, b in pairs:
            if not (0 <= a < pw0) or not (0 <= b < pw0):
                raise ValueError("cross indices must be columns of X_wide")
        for i in range(n):
            for a, b in pairs:
                xw[i] = xw[i] + [xw[i][a] * xw[i][b]]
    pw = len(xw[0])
    pd = len(xd[0])
    widths = [pd] + [int(h) for h in hidden]
    if any(h < 1 for h in widths):
        raise ValueError("hidden layer widths must be positive")
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    lr = float(lr)
    l2 = float(l2)

    state = float(int(seed) % 2147483646) + 1.0
    # MINSTD emits a run of near-zero values from a small seed, which
    # would make every initial weight about -scale and leave the whole
    # ReLU stack dead on arrival.  Discard a fixed warm-up so the seed
    # only picks the stream, not its first magnitudes.
    for _ in range(100):
        state = _lcg(state)

    def _rnd(scale):
        nonlocal state
        state = _lcg(state)
        return (state / _LCG_M - 0.5) * 2.0 * scale

    nlayer = len(widths) - 1
    W = []
    B = []
    for l in range(nlayer):
        fan_in = widths[l]
        scale = np.sqrt(6.0 / (fan_in + widths[l + 1]))
        W.append([[_rnd(scale) for _ in range(fan_in)]
                  for _ in range(widths[l + 1])])
        B.append([0.0] * widths[l + 1])
    w_wide = [0.0] * pw
    w_deep = [0.0] * widths[nlayer]
    bias = 0.0

    def _forward(row_d):
        acts = [row_d]
        a = row_d
        for l in range(nlayer):
            nxt = []
            for u in range(widths[l + 1]):
                s = B[l][u]
                wl = W[l][u]
                for k in range(widths[l]):
                    s += wl[k] * a[k]
                nxt.append(s if s > 0.0 else 0.0)   # eq. (2), ReLU
            acts.append(nxt)
            a = nxt
        return acts

    loss = float("nan")
    for _ep in range(epochs):
        g_wide = [0.0] * pw
        g_deep = [0.0] * widths[nlayer]
        g_bias = 0.0
        gW = [[[0.0] * widths[l] for _ in range(widths[l + 1])]
              for l in range(nlayer)]
        gB = [[0.0] * widths[l + 1] for l in range(nlayer)]
        loss = 0.0
        for i in range(n):
            acts = _forward(xd[i])
            top = acts[nlayer]
            z = bias
            for k in range(pw):
                z += w_wide[k] * xw[i][k]
            for k in range(widths[nlayer]):
                z += w_deep[k] * top[k]
            # eq. (3)
            if z >= 0.0:
                e = np.exp(-z)
                p = 1.0 / (1.0 + e)
                loss += np.log1p(e) + (1.0 - yv[i]) * z
            else:
                e = np.exp(z)
                p = e / (1.0 + e)
                loss += np.log1p(e) - yv[i] * z
            r = p - yv[i]
            g_bias += r
            for k in range(pw):
                g_wide[k] += r * xw[i][k]
            for k in range(widths[nlayer]):
                g_deep[k] += r * top[k]
            # backprop through the ReLU stack
            delta = [r * w_deep[k] for k in range(widths[nlayer])]
            for l in range(nlayer - 1, -1, -1):
                a_prev = acts[l]
                a_cur = acts[l + 1]
                dpre = [delta[u] if a_cur[u] > 0.0 else 0.0
                        for u in range(widths[l + 1])]
                for u in range(widths[l + 1]):
                    du = dpre[u]
                    if du == 0.0:
                        continue
                    gB[l][u] += du
                    gu = gW[l][u]
                    for k in range(widths[l]):
                        gu[k] += du * a_prev[k]
                if l > 0:
                    nd = [0.0] * widths[l]
                    for u in range(widths[l + 1]):
                        du = dpre[u]
                        if du == 0.0:
                            continue
                        wl = W[l][u]
                        for k in range(widths[l]):
                            nd[k] += du * wl[k]
                    delta = nd
        loss = loss / n
        step = lr / n
        bias -= step * g_bias
        for k in range(pw):
            w_wide[k] -= step * (g_wide[k] + l2 * w_wide[k] * n)
        for k in range(widths[nlayer]):
            w_deep[k] -= step * (g_deep[k] + l2 * w_deep[k] * n)
        for l in range(nlayer):
            for u in range(widths[l + 1]):
                B[l][u] -= step * gB[l][u]
                for k in range(widths[l]):
                    W[l][u][k] -= step * (gW[l][u][k] + l2 * W[l][u][k] * n)

    fitted = []
    for i in range(n):
        top = _forward(xd[i])[nlayer]
        z = bias
        for k in range(pw):
            z += w_wide[k] * xw[i][k]
        for k in range(widths[nlayer]):
            z += w_deep[k] * top[k]
        fitted.append(1.0 / (1.0 + np.exp(-z)) if z >= 0.0
                      else np.exp(z) / (1.0 + np.exp(z)))

    return RichResult(
        payload={
            "coef_wide": w_wide,
            "coef_deep": w_deep,
            "bias": float(bias),
            "hidden_weights": W,
            "hidden_bias": B,
            "fitted": fitted,
            "loss": float(loss),
            "n": n,
            "n_wide": pw,
            "n_deep": pd,
            "epochs": epochs,
            "method": "Wide & Deep (Cheng et al. 2016, eqs. 1-3)",
        }
    )


def cheatsheet():
    return "wideD: Wide & Deep jointly trained classifier (Cheng et al. 2016)"


# compact alias per ledger/NAMING.md
wideanddeep = wide_and_deep
