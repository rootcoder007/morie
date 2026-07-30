# morie.fn -- function file (rootcoder007/morie)
"""MINE: mutual information via a neural lower bound."""

import numpy as np

from ._richresult import RichResult

__all__ = ["mine_mutual_information", "mi_neural_estimator"]


def mine_mutual_information(x, y, n_hidden=32, n_iter=600, lr=0.01,
                            seed=0, ema=0.99):
    r"""Donsker-Varadhan lower bound on mutual information.

    .. math::
       I(X;Y) \ge \sup_{T}\;
         \mathbb{E}_{P_{XY}}[T] - \log \mathbb{E}_{P_X P_Y}\big[e^{T}\big]

    with :math:`T` a network and the second expectation taken over
    SHUFFLED pairs, which is what makes the product of marginals
    samplable at all.

    Three properties decide whether the number is usable, and all three
    are returned rather than assumed.

    It is a LOWER BOUND, so it can only understate. An estimate near
    zero means either independence or an inadequate network, and
    nothing in the output distinguishes them -- which is why
    ``bound_note`` says so and why a known-dependent control is the
    only real check.

    The bound is BIASED UPWARD by optimisation. Because the supremum is
    taken over the same data used to evaluate it, the training-set
    value overfits; McAllester and Stratos show any such estimator
    needs sample size exponential in the MI to certify a high value.
    ``holdout_estimate`` evaluates the fitted network on held-out
    pairs, and the gap against the training value is the overfitting.

    The log-mean-exp term has high variance, so the gradient uses the
    exponential moving average correction of the original paper;
    without it training is unstable in exactly the high-MI regime the
    estimator is wanted for.

    Parameters
    ----------
    x, y : array-like, shape (n,) or (n, d)
    n_hidden, n_iter, lr : int, int, float
    seed : int
    ema : float
        Decay for the denominator correction.

    Returns
    -------
    RichResult
        ``mi``, ``mi_nats``, ``mi_bits``, ``holdout_estimate``,
        ``overfit_gap``, ``curve``, ``gaussian_reference``.

    References
    ----------
    Belghazi, Baratin, Rajeshwar, Ozair, Bengio, Courville and
    Hjelm (2018), "MINE: mutual information neural estimation", ICML,
    arXiv:1801.04062.
    McAllester and Stratos (2020), *AISTATS*, on the sample-size limit.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(400, 1))
    >>> out = mine_mutual_information(x, x + rng.normal(size=(400, 1)),
    ...                               n_iter=120)
    >>> bool(out["mi"] > 0)
    True
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    Y = np.atleast_2d(np.asarray(y, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    if Y.shape[0] == 1 and Y.shape[1] > 1:
        Y = Y.T
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("x and y must have the same number of rows.")
    if n < 20:
        raise ValueError("need at least 20 paired observations, got %d." % n)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    d = X.shape[1] + Y.shape[1]
    H = int(n_hidden)

    rng = np.random.default_rng(int(seed))
    cut = max(int(0.8 * n), 10)
    tr = slice(0, cut)
    te = slice(cut, n)
    W1 = rng.normal(scale=np.sqrt(2.0 / d), size=(d, H))
    b1 = np.zeros(H)
    W2 = rng.normal(scale=np.sqrt(2.0 / H), size=(H, 1))
    b2 = np.zeros(1)

    def T(Xa, Ya, W1, b1, W2, b2):
        Z = np.hstack([Xa, Ya])
        A = np.maximum(Z @ W1 + b1, 0.0)
        return (A @ W2 + b2).ravel(), A, Z

    denom_ema = None
    curve = []
    for it in range(int(n_iter)):
        idx = rng.permutation(cut)
        Xj, Yj = X[tr], Y[tr]
        Ym = Yj[idx]
        tj, Aj, Zj = T(Xj, Yj, W1, b1, W2, b2)
        tm, Am, Zm = T(Xj, Ym, W1, b1, W2, b2)
        mx = float(tm.max())
        ex = np.exp(tm - mx)
        mean_ex = float(ex.mean())
        lme = mx + np.log(mean_ex + 1e-300)
        val = float(tj.mean()) - lme
        curve.append(val)

        # EMA-corrected gradient of the log-mean-exp term
        raw = mean_ex * np.exp(mx)
        denom_ema = raw if denom_ema is None else ema * denom_ema + (1 - ema) * raw
        g_j = np.ones_like(tj) / tj.size
        g_m = -np.exp(tm - mx) * np.exp(mx) / (tj.size * max(denom_ema, 1e-300))

        def backprop(g, A, Z):
            gW2 = A.T @ g[:, None]
            gb2 = np.array([g.sum()])
            dA = np.outer(g, W2.ravel()) * (A > 0)
            return Z.T @ dA, dA.sum(0), gW2, gb2

        gW1a, gb1a, gW2a, gb2a = backprop(g_j, Aj, Zj)
        gW1b, gb1b, gW2b, gb2b = backprop(g_m, Am, Zm)
        W1 += lr * (gW1a + gW1b)
        b1 += lr * (gb1a + gb1b)
        W2 += lr * (gW2a + gW2b)
        b2 += lr * (gb2a + gb2b)

    def evaluate(sl):
        Xa, Ya = X[sl], Y[sl]
        if Xa.shape[0] < 5:
            return np.nan
        sh = rng.permutation(Xa.shape[0])
        tj, _, _ = T(Xa, Ya, W1, b1, W2, b2)
        tm, _, _ = T(Xa, Ya[sh], W1, b1, W2, b2)
        mx = float(tm.max())
        return float(tj.mean() - (mx + np.log(np.exp(tm - mx).mean() + 1e-300)))

    mi_tr = evaluate(tr)
    mi_te = evaluate(te)
    # Gaussian reference from the linear correlation, for scale
    ref = np.nan
    if X.shape[1] == 1 and Y.shape[1] == 1:
        r = float(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
        if abs(r) < 1:
            ref = -0.5 * np.log(1 - r ** 2)
    return RichResult(
        payload={
            "estimate": float(mi_tr),
            "mi": float(mi_tr),
            "mi_nats": float(mi_tr),
            "mi_bits": float(mi_tr / np.log(2)),
            "holdout_estimate": float(mi_te),
            "overfit_gap": float(mi_tr - mi_te),
            "overfit_note": (
                "the supremum is taken over the same data it is evaluated "
                "on, so the training value is biased upward; the gap against "
                "the held-out value is that overfitting"
            ),
            "bound_note": (
                "a LOWER bound, so it can only understate; an estimate near "
                "zero means independence OR an inadequate network, and "
                "nothing here distinguishes them"
            ),
            "sample_note": (
                "McAllester and Stratos show any such estimator needs sample "
                "size exponential in the MI to certify a high value, so a "
                "large number on a small sample is not evidence"
            ),
            "gaussian_reference": ref,
            "reference_note": (
                "MI implied by the linear correlation alone; MINE above it "
                "indicates nonlinear dependence, below it indicates the "
                "network has not converged"
            ),
            "curve": np.asarray(curve),
            "converged": bool(len(curve) > 20 and
                              abs(np.mean(curve[-10:])
                                  - np.mean(curve[-20:-10])) < 0.05),
            "n_train": int(cut),
            "n_holdout": int(n - cut),
            "n": int(n),
            "method": "MINE mutual information (Donsker-Varadhan bound)",
        }
    )


def cheatsheet():
    return (
        "miestn: MINE lower bound with a held-out estimate exposing the "
        "optimisation bias, and a Gaussian reference for scale"
    )


#: Catalogue alias for :func:`mine_mutual_information`.
mi_neural_estimator = mine_mutual_information
