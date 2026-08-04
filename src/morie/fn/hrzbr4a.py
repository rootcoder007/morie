# morie.fn -- function file (rootcoder007/morie)
"""Random-coefficients binary-response model and the maximum-score estimator.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 4.1, equations (4.2a)-(4.2d) (page 96), and
Section 4.3.2, equations (4.20)-(4.21) (page 106).

The model is

    Y = 1 if Y* > 0, else 0                                   (4.2a)
    Y* = X'(beta + nu) + V                                    (4.2b)
       = X'beta + (X'nu + V)                                  (4.2c)
       = X'beta + U,     U = X'nu + V                         (4.2d)

so a random-coefficients model is the general binary-response model
with a HETEROSKEDASTIC error.  Mean independence E(U|X) = 0 does not
identify beta (Example 4.1, page 98); median independence does
(Theorem 4.1, page 99) and permits arbitrary heteroskedasticity.

Under median independence median(Y|X=x) = I(x'beta >= 0), so beta
minimises S_bin(b) = E|Y - I(X'b >= 0)| (4.17), whose sample analog
(4.20) reduces to

    maximize_{|b1| = 1, b}  S_ms(b) = (1/n) sum_i (2Y_i - 1)
                                      I(X_i'b >= 0)            (4.21)

Any solution is a maximum-score estimator.  Its rate is n^{-1/3} and
its limit is NOT normal, so standard errors do not give confidence
intervals (page 108); the smoothed estimator is what restores them.

(4.21) is a step function of b, so it is maximised here by exact
enumeration of the sign patterns the data can produce: with one free
coefficient the breakpoints are b2 = -X_i1 / X_i2 and the objective is
evaluated at the midpoint of each adjacent pair.  With more free
coefficients a fixed deterministic grid is used.  Nothing is random and
nothing exits on a tolerance.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["binresp", "horowitz_binary_response_model"]


def _score(X, yv, b):
    return float(np.sum((2.0 * yv - 1.0)
                        * ((X @ b) >= 0.0).astype(float))) / X.shape[0]


def binresp(x, y, ngrid=41, blim=5.0):
    """Maximum-score estimator for the random-coefficients binary model.

    Parameters
    ----------
    x : array-like, (n, d)
        Covariates; the first column carries the scale normalisation
        |beta_1| = 1 and should be continuously distributed.
    y : array-like, (n,) of 0/1
    ngrid : int, default 41
        Grid points per free coefficient when d > 2.  Fixed.
    blim : float, default 5.0
        Half-width of that grid.

    Returns
    -------
    RichResult
        payload keys: estimate, score, ncand, correct, rate, limit,
        seusable, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    uy = np.unique(yv)
    if bool(np.any((uy != 0.0) & (uy != 1.0))):
        raise ValueError("y must be binary 0/1 for a binary-response model.")
    if d < 2:
        raise ValueError("need at least two covariates for a scale normalisation.")

    if d == 2:
        nz = np.abs(X[:, 1]) > 1e-12
        cuts = np.sort(-X[nz, 0] / X[nz, 1])
        cand = [[float(cuts[0]) - 1.0]] if cuts.size else [[0.0]]
        for k in range(int(cuts.size) - 1):
            cand.append([0.5 * (float(cuts[k]) + float(cuts[k + 1]))])
        if cuts.size:
            cand.append([float(cuts[-1]) + 1.0])
    else:
        axis = np.linspace(-float(blim), float(blim), int(ngrid))
        cand = [[0.0] * (d - 1)]
        for j in range(d - 1):
            new = []
            for base in cand:
                for v in axis:
                    nb = list(base)
                    nb[j] = float(v)
                    new.append(nb)
            cand = new

    best = None
    bestval = -1e300
    for c in cand:
        b = np.concatenate([np.array([1.0]), np.asarray(c, dtype=float)])
        v = _score(X, yv, b)
        if v > bestval:
            bestval = v
            best = b
    pred = (X @ best >= 0.0).astype(float)
    correct = float(np.mean((pred == yv).astype(float)))
    return RichResult(
        title="Maximum-score estimator, random-coefficients binary response",
        payload={"estimate": best, "score": float(bestval),
                 "ncand": int(len(cand)), "correct": correct,
                 "rate": -1.0 / 3.0, "limit": "nonnormal",
                 "seusable": False, "n": n,
                 "method": "Horowitz (2009) eq. (4.2), (4.21) maximum score"},
    )


horowitz_binary_response_model = binresp


def cheatsheet():
    return "hrzbr4a: random-coefficients binary response, maximum-score estimator"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 200
    X = np.column_stack([np.linspace(-2, 2, n),
                         np.cos(np.arange(1, n + 1) * 0.8)])
    yv = ((X @ np.array([1.0, 0.6])) >= 0.0).astype(float)
    r = binresp(X, yv)
    assert r["correct"] == 1.0, r["correct"]
    assert abs(float(r["estimate"][0]) - 1.0) < 1e-12
    assert not r["seusable"]
    print("ok", r["estimate"], r["score"])
