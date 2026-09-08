"""MICE multiple imputation by chained equations (van Buuren 2018)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["miord2", "mice_chained_equations"]


def _inv(a):
    k = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(k)]
         for i, row in enumerate(a)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise ValueError("singular matrix in norm draw")
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        for j in range(2 * k):
            m[c][j] /= d
        for r in range(k):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                for j in range(2 * k):
                    m[r][j] -= f * m[c][j]
    return [row[k:] for row in m]


def _chol(a):
    k = len(a)
    l = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            s = sum(l[i][t] * l[j][t] for t in range(j))
            if i == j:
                v = a[i][i] - s
                l[i][j] = math.sqrt(max(v, 1e-300))
            else:
                l[i][j] = (a[i][j] - s) / l[j][j]
    return l


def _norm_draw(rng, x_obs, y_obs, x_mis, kappa):
    # Algorithm 3.1 of van Buuren (2018): Bayesian normal linear draw.
    n1 = len(y_obs)
    q = len(x_obs[0])
    s = [[0.0] * q for _ in range(q)]
    xty = [0.0] * q
    for xi, yi in zip(x_obs, y_obs):
        for r in range(q):
            xty[r] += xi[r] * yi
            for c in range(q):
                s[r][c] += xi[r] * xi[c]
    a = [[s[r][c] + (kappa * s[r][r] if r == c else 0.0)
          for c in range(q)] for r in range(q)]
    v = _inv(a)
    beta_hat = [sum(v[r][c] * xty[c] for c in range(q)) for r in range(q)]
    ssr = 0.0
    for xi, yi in zip(x_obs, y_obs):
        e = yi - sum(b * t for b, t in zip(beta_hat, xi))
        ssr += e * e
    nu = max(n1 - q, 1)
    g = 2.0 * float(rng.gamma(nu / 2.0))          # chi^2_nu draw
    sigma = math.sqrt(ssr / max(g, 1e-300))
    z1 = [float(rng.normal()) for _ in range(q)]
    l = _chol(v)
    beta_dot = [beta_hat[r] + sigma * sum(l[r][t] * z1[t]
                                          for t in range(r + 1))
                for r in range(q)]
    out = []
    for xi in x_mis:
        z2 = float(rng.normal())
        out.append(sum(b * t for b, t in zip(beta_dot, xi)) + sigma * z2)
    return out


def miord2(data, m=5, maxit=5, seed=0, kappa=1e-4):
    """
    Multivariate imputation by chained equations (MICE), normal model.

    Implements Algorithm 4.3 of van Buuren (2018) with the Bayesian
    normal linear imputation model of his Algorithm 3.1 (the mice
    method "norm", adapted from Rubin 1987, p. 167): starting
    imputations are random draws from the observed values of each
    variable; then for M iterations each incomplete variable Y_j is
    regressed on the currently-complete other variables, the
    parameters (beta, sigma) are drawn from their posterior
    (V = (S + diag(S) kappa)^{-1}, sigma^2 = SSR/chi^2_{n1-q},
    beta_dot = beta_hat + sigma L z1 with L the Cholesky factor of
    V), and the missing entries are drawn as X_mis beta_dot +
    sigma z2.  Executing the chain m times in parallel yields m
    completed data sets; being a Gibbs-type sampler, imputations are
    "proper" in Rubin's sense.

    Sources
    -------
    van Buuren, S. (2018). *Flexible Imputation of Missing Data*,
    2nd ed., Chapman & Hall/CRC, Algorithm 4.3 (MICE) and Algorithm
    3.1 (Bayesian normal draw) (local copies
    fetched-wave3/vanbuuren-fimd-ch4-mice.html,
    fetched-wave3/vanbuuren-fimd-ch3-norm.html).
    van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice:
    Multivariate imputation by chained equations in R. *Journal of
    Statistical Software*, 45(3).
    Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in
    Surveys*, Wiley, p. 167.

    Parameters
    ----------
    data : sequence of rows
        Rectangular data; missing entries are None (or nan).  All
        variables treated as continuous.
    m : int
        Number of imputed data sets.
    maxit : int
        Iterations per chain (van Buuren: 5-10 usually suffice).
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).
    kappa : float
        Ridge parameter of Algorithm 3.1 (default 1e-4).

    Returns
    -------
    RichResult
        Keys: imputations (m completed data sets), missing_mask,
        m, maxit, column_means (per imputation).
    """
    rows = [[(None if v is None or (isinstance(v, float)
                                    and v != v) else float(v))
             for v in r] for r in data]
    n = len(rows)
    if n < 3:
        raise ValueError("need at least three rows")
    p = len(rows[0])
    if any(len(r) != p for r in rows):
        raise ValueError("rows must have equal length")
    mask = [[rows[i][j] is None for j in range(p)] for i in range(n)]
    mis_cols = [j for j in range(p)
                if any(mask[i][j] for i in range(n))]
    for j in range(p):
        if all(mask[i][j] for i in range(n)):
            raise ValueError("column %d has no observed values" % j)
    m = int(m)
    maxit = int(maxit)
    rng = np.random.default_rng(seed)
    imps = []
    for _chain in range(m):
        cur = [r[:] for r in rows]
        # starting imputations: random draws from observed values
        for j in mis_cols:
            obs = [rows[i][j] for i in range(n) if not mask[i][j]]
            for i in range(n):
                if mask[i][j]:
                    pick = min(int(float(rng.uniform()) * len(obs)),
                               len(obs) - 1)
                    cur[i][j] = obs[pick]
        for _t in range(maxit):
            for j in mis_cols:
                x_obs, y_obs, x_mis, mis_idx = [], [], [], []
                for i in range(n):
                    xi = [1.0] + [cur[i][c] for c in range(p) if c != j]
                    if mask[i][j]:
                        x_mis.append(xi)
                        mis_idx.append(i)
                    else:
                        x_obs.append(xi)
                        y_obs.append(rows[i][j])
                if not mis_idx:
                    continue
                draws = _norm_draw(rng, x_obs, y_obs, x_mis, kappa)
                for i, v in zip(mis_idx, draws):
                    cur[i][j] = v
        imps.append(cur)
    means = [[sum(imp[i][j] for i in range(n)) / n for j in range(p)]
             for imp in imps]
    return RichResult(payload={
        "imputations": imps,
        "missing_mask": mask,
        "m": m,
        "maxit": maxit,
        "column_means": means,
        "seed": int(seed),
        "method": "MICE norm (van Buuren Algs. 3.1 + 4.3)",
    })


# long descriptive alias (stub-era name)
mice_chained_equations = miord2


def cheatsheet():
    return "miord2: chained Bayesian-normal imputation, m chains x maxit sweeps"

# public names resolved by fn/_lazy_map.json
mi_chained_eq = miord2
michainedeq = miord2
