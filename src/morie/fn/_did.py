# morie.fn -- shared helpers (rootcoder007/morie)
"""Panel, weighting and estimation primitives for the DiD shelf.

Everything in the difference-in-differences literature since about
2018 is a response to one discovery: the two-way fixed-effects
regression that the field used as *the* DiD estimator does not
estimate an average treatment effect when treatment timing varies and
effects are heterogeneous. It estimates a weighted sum of every
possible 2x2 comparison in the data, INCLUDING comparisons that use
already-treated units as controls, and the weights on those can be
negative.

The helpers here are the pieces the modern estimators share:

* a strict panel builder, because every result below assumes a
  balanced, absorbing design and silently returns nonsense otherwise;
* two-way demeaning, which makes the TWFE coefficient a one-line
  inner product and is the object Goodman-Bacon decomposes;
* a simplex-constrained least-squares solver, for the synthetic
  control and synthetic DiD weights;
* propensity and outcome fits for the doubly-robust estimators.
"""

from . import _array_core as np

__all__ = [
    "as_panel",
    "first_treatment",
    "demean_two_way",
    "twfe_beta",
    "cluster_se",
    "simplex_project",
    "simplex_lstsq",
    "logit_fit",
    "logit_predict",
    "ols_fit",
    "add_intercept",
]


def as_panel(y, unit, time):
    """Reshape long data into a balanced unit-by-period matrix.

    Refuses an unbalanced or duplicated panel rather than filling
    gaps. Every estimator on this shelf differences a unit against
    its own earlier value; a missing cell is a missing comparison,
    not a value to impute quietly.

    Returns
    -------
    Y : ndarray, shape (n_units, n_periods)
    units, periods : ndarray
        Sorted unique identifiers, giving the row and column order.
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(unit).ravel()
    t = np.asarray(time).ravel()
    if not (len(y) == len(u) == len(t)):
        raise ValueError(
            "y, unit and time must have the same length, got %d, %d and %d."
            % (len(y), len(u), len(t))
        )
    units, ui = np.unique(u, return_inverse=True)
    periods, ti = np.unique(t, return_inverse=True)
    n, T = len(units), len(periods)
    if n < 2 or T < 2:
        raise ValueError(
            "need at least 2 units and 2 periods, got %d and %d." % (n, T)
        )
    Y = np.full((n, T), np.nan)
    if np.any(np.bincount(ui * T + ti, minlength=n * T) > 1):
        raise ValueError("the panel has duplicate (unit, time) observations.")
    Y[ui, ti] = y
    if np.isnan(Y).any():
        miss = int(np.isnan(Y).sum())
        raise ValueError(
            "the panel is unbalanced: %d of %d unit-period cells are absent. "
            "Every estimator here differences a unit against its own earlier "
            "value, so a missing cell is a missing comparison." % (miss, n * T)
        )
    return Y, units, periods


def first_treatment(D, unit, time, units=None, periods=None):
    """Cohort (first-treated period index) per unit, ``inf`` if never.

    Treatment must be absorbing -- once on, never off. Staggered-DiD
    identification is defined for adoption, not for switching in and
    out, and every aggregation below indexes time relative to a
    single adoption date.
    """
    Dm, u2, p2 = as_panel(D, unit, time)
    if units is not None and not np.array_equal(u2, np.asarray(units)):
        raise ValueError("the treatment panel has a different unit set.")
    if periods is not None and not np.array_equal(p2, np.asarray(periods)):
        raise ValueError("the treatment panel has a different period set.")
    if not np.all(np.isin(Dm, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1.")
    off_after_on = np.any(np.diff(Dm, axis=1) < 0, axis=1)
    if off_after_on.any():
        bad = u2[off_after_on][:5]
        raise ValueError(
            "treatment must be absorbing; unit(s) %s switch back to untreated. "
            "Staggered-DiD identification is defined for adoption, not for "
            "switching in and out." % (list(bad),)
        )
    g = np.full(Dm.shape[0], np.inf)
    ever = Dm.any(axis=1)
    g[ever] = np.argmax(Dm[ever] > 0, axis=1).astype(float)
    return g, Dm, u2, p2


def demean_two_way(M):
    """Subtract unit and period means and add the grand mean back.

    On a balanced panel this is the exact two-way within transform in
    one pass -- the Frisch-Waugh residual of regressing on unit and
    period dummies.
    """
    M = np.asarray(M, dtype=float)
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def twfe_beta(Y, Dm):
    """The two-way fixed-effects coefficient on treatment.

    After two-way demeaning the estimator is one inner product; that
    ratio is the object Goodman-Bacon (2021) decomposes into 2x2
    comparisons.
    """
    Yt = demean_two_way(Y)
    Dt = demean_two_way(Dm)
    denom = float(np.sum(Dt * Dt))
    if denom <= 0:
        raise ValueError(
            "treatment has no within-panel variation after removing unit and "
            "period effects; no DiD comparison exists."
        )
    beta = float(np.sum(Dt * Yt) / denom)
    resid = Yt - beta * Dt
    return beta, resid, Dt, denom


def cluster_se(resid, Dt, denom, n_units):
    """Unit-clustered standard error for the TWFE coefficient.

    Clustering is on the unit, because serial correlation within a
    unit is the failure Bertrand, Duflo and Mullainathan (2004)
    showed inflates DiD t-statistics severalfold.
    """
    scores = np.sum(Dt * resid, axis=1)
    meat = float(np.sum(scores**2))
    correction = n_units / max(n_units - 1.0, 1.0)
    return float(np.sqrt(correction * meat) / denom)


def simplex_project(v):
    """Euclidean projection onto the unit simplex (Duchi et al. 2008)."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def simplex_lstsq(A, b, zeta=0.0, intercept=False, max_iter=5000, tol=1e-12):
    """Minimise ``||A w + c - b||^2 + zeta^2 n ||w||^2`` over the simplex.

    Non-negative weights summing to one are what make a synthetic
    control a WEIGHTED AVERAGE of real units rather than an
    extrapolation, so the constraint is the method, not a numerical
    convenience. ``zeta`` is the ridge term of Arkhangelsky et al.
    (2021), which spreads weight across donors instead of letting a
    handful of units carry it; ``intercept`` allows the synthetic
    unit to sit at a constant offset, which synthetic DiD permits and
    classical synthetic control does not.

    Solved by projected gradient with backtracking -- the objective is
    convex and the simplex projection is exact, so the iteration is
    deterministic and needs no solver dependency.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if A.shape[0] != b.size:
        raise ValueError(
            "A has %d rows and b has %d entries." % (A.shape[0], b.size)
        )
    m, n = A.shape
    reg = float(zeta) ** 2 * m
    w = np.full(n, 1.0 / n)
    c = 0.0

    def obj(w, c):
        r = A @ w + c - b
        return float(r @ r + reg * (w @ w))

    lip = float(np.linalg.norm(A, 2) ** 2 + reg)
    step = 1.0 / max(lip, 1e-12)
    prev = obj(w, c)
    for _ in range(int(max_iter)):
        r = A @ w + c - b
        grad = 2.0 * (A.T @ r + reg * w)
        s = step
        for _ in range(60):
            wn = simplex_project(w - s * grad)
            cn = float(np.mean(b - A @ wn)) if intercept else 0.0
            if obj(wn, cn) <= prev:
                break
            s *= 0.5
        cur = obj(wn, cn)
        moved = float(np.max(np.abs(wn - w)))
        w, c = wn, cn
        if prev - cur < tol * max(1.0, abs(prev)) and moved < 1e-10:
            prev = cur
            break
        prev = cur
    return w, c, prev


def add_intercept(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    if np.any(np.all(np.isclose(X, 1.0), axis=0)):
        return X
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_fit(X, y, ridge=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def logit_fit(X, y, max_iter=100, tol=1e-10, ridge=1e-8):
    """Logistic regression by IRLS with a whisper of ridge.

    The ridge term is there only so a perfectly separated propensity
    -- common when a covariate predicts adoption exactly -- returns
    finite coefficients instead of diverging. Separation is a real
    warning about overlap, and the caller is told about it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    beta = np.zeros(X.shape[1])
    separated = False
    for _ in range(int(max_iter)):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        wgt = np.maximum(p * (1 - p), 1e-10)
        z = eta + (y - p) / wgt
        XtW = X.T * wgt
        step = np.linalg.solve(XtW @ X + ridge * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(step - beta)) < tol:
            beta = step
            break
        beta = step
    p = logit_predict(X, beta)
    if np.min(p) < 1e-6 or np.max(p) > 1 - 1e-6:
        separated = True
    return beta, separated


def logit_predict(X, beta):
    eta = np.clip(np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float),
                  -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))
