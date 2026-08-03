# morie.fn -- genomic-prediction core (rootcoder007/morie)
"""Shared machinery for the MVSML shelf.

Everything follows Montesinos López, Montesinos López & Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (open access, DOI 10.1007/978-3-030-89010-0) --
equations checked against the library PDF:

* one-way models, eq. (1.2)-(1.5) pp.15-16;
* mixed model + Henderson MME, eq. (2.1)-(2.2) p.36;
* GBLUP MME eq. (2.3) p.53 and SNP-BLUP MME eq. (2.4) p.53;
* genomic relationship matrix method 3, p.52
  (``G = scale(X) scale(X)' / ncol(X)``);
* PCA compression, sec. 2.8 pp.63-64 (``PC = XW``, ``X* = XW*``);
* OLS/ML/GD for the linear model, eq. (3.1) p.71 and pp.72-79;
* expected prediction error, p.80; ridge PRSS and solution, p.81;
* classification metrics eq. (4.5)-(4.14) pp.131-136.
"""

from __future__ import annotations

import math

from . import _array_core as np


def _flat(x):
    if hasattr(x, "_flat"):
        return [float(v) for v in x._flat()]
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _mat(A):
    return [[float(v) for v in row] for row in A]


def _t(A):
    return [list(col) for col in zip(*A)]


def _mm(A, B):
    Bt = _t(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt]
            for row in A]


def _mv(A, v):
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def _solve(A, b):
    """Solve A x = b through the native linear-algebra core.

    Rank-deficient systems are common here and are not an error: an
    intercept-only design block, a covariate column of zeros, or a
    genomic relationship matrix built from fewer markers than lines all
    produce a singular left-hand side.  Rather than failing, fall back
    on the rank-gated pseudo-inverse (``linalg.pinv``, the same
    statsmodels pinv-of-design path used by the OLS/GLM solver in this
    package), which returns the minimum-norm solution.
    """
    Am = _mat(A)
    bv = _flat(b)
    try:
        x = np.linalg.solve(np.marr(Am), np.marr(bv))
        return [float(v) for v in x._flat()]
    except Exception:
        P = np.linalg.pinv(np.marr(Am))
        Pl = [[float(v) for v in row] for row in P._tolist()] \
            if hasattr(P, "_tolist") else _mat(P)
        return _mv(Pl, bv)


def _inv(A):
    n = len(A)
    cols = [_solve(A, [1.0 if r == c else 0.0 for r in range(n)])
            for c in range(n)]
    return _t(cols)


# ------------------------------------------- one-way models (ch. 1)
def one_way_summary(groups):
    """Fits the three chapter-1 models to a balanced one-way layout.

    ``groups`` is a list of equal-length observation lists (one per
    level).  Returns the single-mean fit of eq. (1.2), the fixed-effects
    fit of eq. (1.3) and the random-effects fit of eq. (1.5); the
    reparameterization of eq. (1.4) is the pair (grand mean, deviations).
    Variance components use the balanced ANOVA identities
    ``sigma2_e = MSE`` and ``sigma2_b = (MSB - MSE) / r``.
    """
    gs = [[float(v) for v in g] for g in groups]
    if not gs or any(len(g) != len(gs[0]) for g in gs):
        raise ValueError("need a balanced layout (equal group sizes)")
    a = len(gs)
    r = len(gs[0])
    n = a * r
    means = [sum(g) / r for g in gs]
    grand = sum(sum(g) for g in gs) / n
    ss_between = r * sum((m - grand) ** 2 for m in means)
    ss_within = sum((v - m) ** 2 for g, m in zip(gs, means) for v in g)
    ms_between = ss_between / (a - 1)
    ms_within = ss_within / (n - a)
    sigma2_b = max((ms_between - ms_within) / r, 0.0)
    return {
        "grand_mean": grand,                       # eq. (1.2) beta-hat
        "sd_single_mean": math.sqrt(
            (ss_between + ss_within) / (n - 1)),   # eq. (1.2) sigma-hat
        "group_means": means,                      # eq. (1.3) beta_i
        "sd_residual": math.sqrt(ms_within),       # eq. (1.3)/(1.5)
        "deviations": [m - grand for m in means],  # eq. (1.4)
        "sigma2_b": sigma2_b,                      # eq. (1.5)
        "icc": sigma2_b / (sigma2_b + ms_within)
        if sigma2_b + ms_within > 0 else 0.0,
        "ms_between": ms_between, "ms_within": ms_within,
    }


# --------------------------------------- mixed models / MME (ch. 2)
def mme_solve(X, Z, y, Sigma_inv, R_inv=None):
    """Henderson's mixed model equations, eq. (2.2) p.36:

        [X'R^-1 X   X'R^-1 Z          ] [beta] = [X'R^-1 y]
        [Z'R^-1 X   Z'R^-1 Z + Sig^-1 ] [ u  ]   [Z'R^-1 y]

    ``R_inv`` defaults to the identity (homoscedastic residuals).
    Returns (beta-hat = BLUE, u-hat = BLUP).
    """
    X = _mat(X)
    Z = _mat(Z)
    y = _flat(y)
    n = len(y)
    p = len(X[0])
    q = len(Z[0])
    if R_inv is None:
        R_inv = [[1.0 if i == j else 0.0 for j in range(n)]
                 for i in range(n)]
    else:
        R_inv = _mat(R_inv)
    Xt = _t(X)
    Zt = _t(Z)
    XtRi = _mm(Xt, R_inv)
    ZtRi = _mm(Zt, R_inv)
    A11 = _mm(XtRi, X)
    A12 = _mm(XtRi, Z)
    A21 = _mm(ZtRi, X)
    A22 = _mm(ZtRi, Z)
    S = _mat(Sigma_inv)
    for i in range(q):
        for j in range(q):
            A22[i][j] += S[i][j]
    LHS = [A11[i] + A12[i] for i in range(p)] + \
        [A21[i] + A22[i] for i in range(q)]
    RHS = _mv(XtRi, y) + _mv(ZtRi, y)
    sol = _solve(LHS, RHS)
    return sol[:p], sol[p:]


def blue_blup_via_v(X, Z, y, Sigma, R=None):
    """The equivalent V-based solution given below eq. (2.2) p.36:
    V = Z Sigma Z' + R, beta = (X'V^-1 X)^-1 X'V^-1 y,
    u = Sigma Z' V^-1 (y - X beta)."""
    X = _mat(X)
    Z = _mat(Z)
    y = _flat(y)
    n = len(y)
    if R is None:
        R = [[1.0 if i == j else 0.0 for j in range(n)]
             for i in range(n)]
    V = _mm(_mm(Z, _mat(Sigma)), _t(Z))
    for i in range(n):
        for j in range(n):
            V[i][j] += R[i][j]
    Vi = _inv(V)
    Xt = _t(X)
    XtVi = _mm(Xt, Vi)
    beta = _solve(_mm(XtVi, X), _mv(XtVi, y))
    resid = [yi - v for yi, v in zip(y, _mv(X, beta))]
    u = _mv(_mm(_mat(Sigma), _t(Z)), _mv(Vi, resid))
    return beta, u


def grm_vanraden_method3(M):
    """Genomic relationship matrix, method 3 of sec. 2.4 as used on
    p.52: scale the marker matrix by column (centre and divide by the
    sample standard deviation) and set G = Xs Xs' / ncol(Xs)."""
    Mm = _mat(M)
    n = len(Mm)
    p = len(Mm[0])
    cols = _t(Mm)
    scaled = []
    for c in cols:
        mu = sum(c) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in c) / (n - 1))
        scaled.append([(v - mu) / sd if sd > 0 else 0.0 for v in c])
    Xs = _t(scaled)
    G = _mm(Xs, _t(Xs))
    return [[v / p for v in row] for row in G]


def gblup_gebv(X, y, G, sigma2_g, sigma2_e=1.0, use_mme=False):
    """GBLUP of eq. (2.3) p.53: Z = I over the lines and
    Sigma = sigma2_g G, so the MME block carries G^-1/sigma2_g.

    A genomic relationship matrix built from p markers has rank at
    most p, so G is singular whenever the number of lines exceeds the
    number of markers and the G^-1 form of (2.3) cannot be used.  The
    default path is therefore the equivalent solution given directly
    below eq. (2.2) on p.36 -- beta = (X'V^-1X)^-1X'V^-1y and
    u = Sigma Z'V^-1(y - X beta) with V = Z Sigma Z' + R -- which needs
    no inverse of G.  Pass ``use_mme=True`` to solve the literal (2.3)
    system when G is known to be nonsingular.
    """
    q = len(G)
    n = len(_flat(y))
    Z = [[1.0 if i == j else 0.0 for j in range(q)] for i in range(n)]
    Sigma = [[sigma2_g * G[i][j] for j in range(q)] for i in range(q)]
    if use_mme:
        Gi = _inv(_mat(G))
        Sig_inv = [[Gi[i][j] * sigma2_e / sigma2_g for j in range(q)]
                   for i in range(q)]
        return mme_solve(X, Z, y, Sig_inv)
    R = [[sigma2_e if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    return blue_blup_via_v(X, Z, y, Sigma, R)


def snp_blup_gebv(X, y, M, sigma2_m, sigma2_e=1.0):
    """SNP-BLUP through eq. (2.4) p.53: Z = M (scaled markers) and
    Sigma = sigma2_M I.  GEBV = M u-hat."""
    Mm = _mat(M)
    p = len(Mm[0])
    n = len(_flat(y))
    Sigma = [[sigma2_m if i == j else 0.0 for j in range(p)]
             for i in range(p)]
    R = [[sigma2_e if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    beta, u = blue_blup_via_v(X, Mm, y, Sigma, R)
    return beta, u, _mv(Mm, u)


def scale_columns(M, center=True, scale=True):
    """R's ``scale()``: centre by the column mean and divide by the
    column sample standard deviation (denominator n - 1)."""
    Mm = _mat(M)
    n = len(Mm)
    out_cols = []
    for c in _t(Mm):
        mu = sum(c) / n if center else 0.0
        if scale:
            sd = math.sqrt(sum((v - mu) ** 2 for v in c) / (n - 1))
        else:
            sd = 1.0
        out_cols.append([(v - mu) / sd if sd > 0 else 0.0 for v in c])
    return _t(out_cols)


# ------------------------------------------------- PCA (sec. 2.8)
def pca_compress(X, k=None, use_correlation=True):
    """sec. 2.8 pp.63-64: Q = X'X/(n-1) on scaled columns, W the
    eigenvectors of Q, PC = XW, and the compressed matrix
    X* = XW* keeping the first k columns of W."""
    Xs = scale_columns(X) if use_correlation else _mat(X)
    n = len(Xs)
    p = len(Xs[0])
    Q = _mm(_t(Xs), Xs)
    Q = [[v / (n - 1) for v in row] for row in Q]
    vals, vecs = np.linalg.eigh(np.marr(Q))
    lam = [float(v) for v in vals._flat()]
    W = [[float(v) for v in row] for row in vecs._tolist()] \
        if hasattr(vecs, "_tolist") else _mat(vecs)
    order = sorted(range(p), key=lambda i: -lam[i])
    lam = [lam[i] for i in order]
    W = _t([[W[r][i] for r in range(p)] for i in order])
    PC = _mm(Xs, W)
    tot = sum(lam)
    k = p if k is None else int(k)
    return {
        "eigenvalues": lam,
        "sd_pc": [math.sqrt(max(v, 0.0)) for v in lam],
        "loadings": W,
        "scores": PC,
        "compressed": [row[:k] for row in PC],
        "prop_variance": [v / tot for v in lam],
        "cum_variance": [sum(lam[:i + 1]) / tot for i in range(p)],
    }


# --------------------------------- linear model (ch. 3 pp.71-82)
def ols_fit(X, y, add_intercept=False):
    """eq. (3.1) p.71 fitted by least squares: beta = (X'X)^-1 X'y,
    sigma2 = RSS/(n - p - 1), Var(beta) = sigma2 (X'X)^-1, hat matrix
    H = X(X'X)^-1X' (pp.72-73)."""
    Xm = _mat(X)
    if add_intercept:
        Xm = [[1.0] + row for row in Xm]
    y = _flat(y)
    n = len(y)
    p1 = len(Xm[0])
    XtX = _mm(_t(Xm), Xm)
    beta = _solve(XtX, _mv(_t(Xm), y))
    fitted = _mv(Xm, beta)
    resid = [a - b for a, b in zip(y, fitted)]
    rss = sum(v * v for v in resid)
    dof = n - p1
    sigma2 = rss / dof if dof > 0 else float("nan")
    XtXi = _inv(XtX)
    return {
        "beta": beta, "fitted": fitted, "residuals": resid,
        "rss": rss, "sigma2": sigma2,
        "sigma2_ml": rss / n,                    # p.75 ML estimate
        "var_beta": [[sigma2 * XtXi[i][j] for j in range(p1)]
                     for i in range(p1)],
        "se_beta": [math.sqrt(sigma2 * XtXi[i][i]) for i in range(p1)],
    }


def gradient_descent_ols(X, y, alpha=1e-2, tol=1e-8, max_iter=100000,
                         optimal_step=False, add_intercept=False):
    """sec. 3.4 p.76: eta_{t+1} = eta_t - alpha grad f(eta_t) for the
    residual sum of squares; with ``optimal_step`` the exact line-search
    step of the book's R code is used,
    alpha = (e'X X'e) / (e'X (X'X) X'e)."""
    Xm = _mat(X)
    if add_intercept:
        Xm = [[1.0] + row for row in Xm]
    y = _flat(y)
    p = len(Xm[0])
    beta = [0.0] * p
    beta[0] = sum(y) / len(y) if add_intercept else 0.0
    Xt = _t(Xm)
    XtX = _mm(Xt, Xm)
    it = 0
    gap = 1.0
    while gap > tol and it < max_iter:
        it += 1
        e = [a - b for a, b in zip(y, _mv(Xm, beta))]
        g = _mv(Xt, e)                       # X'e = -0.5 grad RSS
        if optimal_step:
            num = sum(v * v for v in g)
            den = sum(a * b for a, b in zip(g, _mv(XtX, g)))
            step = num / den if den > 0 else alpha
        else:
            step = alpha
        new = [b + step * gi for b, gi in zip(beta, g)]
        gap = max(abs(a - b) for a, b in zip(new, beta))
        beta = new
    return {"beta": beta, "iterations": it, "tolerance": gap}


def expected_prediction_error(sigma2, x_star, eigenvalues):
    """p.80: EPE(x_o) = sigma2 (1 + sum_j (x*_oj)^2 / lambda_j), for
    x* = Gamma' x_o the rotated feature vector and lambda_j the
    eigenvalues of X'X."""
    xs = _flat(x_star)
    lam = _flat(eigenvalues)
    if any(v <= 0 for v in lam):
        raise ValueError("eigenvalues must be positive")
    return float(sigma2) * (1.0 + sum(v * v / l
                                      for v, l in zip(xs, lam)))


def ridge_fit(X, y, lam, add_intercept=True):
    """sec. 3.6.1 p.81: PRSS_lambda(beta) = RSS(beta) + lambda beta'D beta
    with D = diag(0, 1, ..., 1); beta^R(lambda) = (X'X + lambda D)^-1 X'y.
    The intercept column is never penalized."""
    Xm = _mat(X)
    if add_intercept:
        Xm = [[1.0] + row for row in Xm]
    y = _flat(y)
    p = len(Xm[0])
    A = _mm(_t(Xm), Xm)
    for j in range(p):
        if not (add_intercept and j == 0):
            A[j][j] += float(lam)
    beta = _solve(A, _mv(_t(Xm), y))
    fitted = _mv(Xm, beta)
    resid = [a - b for a, b in zip(y, fitted)]
    rss = sum(v * v for v in resid)
    pen = float(lam) * sum(b * b for j, b in enumerate(beta)
                           if not (add_intercept and j == 0))
    return {"beta": beta, "fitted": fitted, "rss": rss,
            "penalty": pen, "prss": rss + pen}


# ------------------------------- classification metrics (ch. 4)
def confusion_counts(y_true, y_pred, n_classes=None):
    """The C x C confusion matrix of Table 4.3 (rows observed,
    columns predicted)."""
    yt = [int(v) for v in _flat(y_true)]
    yp = [int(v) for v in _flat(y_pred)]
    C = n_classes or (max(yt + yp) + 1)
    M = [[0 for _ in range(C)] for _ in range(C)]
    for a, b in zip(yt, yp):
        M[a][b] += 1
    return M


def class_metrics(conf, i):
    """eq. (4.5)-(4.12) pp.131-132 on a one-versus-all basis:
    TFN_i = sum_{j != i} n_ij, TFP_i = sum_{j != i} n_ji,
    TTN_i = sum_{j != i} sum_{k != i} n_jk, TTP_all = sum_j n_jj,
    P_i = TTP/(TTP + TFP_i), Se_i = TTP/(TTP + TFN_i),
    Sp_i = TTN_i/(TTN_i + TFP_i), pCCC = TTP / sum_ij n_ij."""
    C = len(conf)
    tfn = sum(conf[i][j] for j in range(C) if j != i)
    tfp = sum(conf[j][i] for j in range(C) if j != i)
    ttn = sum(conf[j][k] for j in range(C) for k in range(C)
              if j != i and k != i)
    ttp = sum(conf[j][j] for j in range(C))
    total = sum(sum(row) for row in conf)
    return {
        "TFN": tfn, "TFP": tfp, "TTN": ttn, "TTP_all": ttp,
        "precision": ttp / (ttp + tfp) if ttp + tfp else 0.0,
        "sensitivity": ttp / (ttp + tfn) if ttp + tfn else 0.0,
        "specificity": ttn / (ttn + tfp) if ttn + tfp else 0.0,
        "pCCC": ttp / total if total else 0.0,
    }


def binary_metrics(y_true, y_pred, positive=1):
    """Table 4.4 p.133 with the two-class reductions: PCCC = (tp+tn)/n,
    Se = tp/(tp+fn), Sp = tn/(tn+fp), P = tp/(tp+fp), and Cohen's
    kappa = (P0 - Pe)/(1 - Pe) with
    Pe = (tp+fn)/n (tp+fp)/n + (fp+tn)/n (fn+tn)/n."""
    yt = [int(v) for v in _flat(y_true)]
    yp = [int(v) for v in _flat(y_pred)]
    pos = int(positive)
    tp = sum(1 for a, b in zip(yt, yp) if a == pos and b == pos)
    tn = sum(1 for a, b in zip(yt, yp) if a != pos and b != pos)
    fp = sum(1 for a, b in zip(yt, yp) if a != pos and b == pos)
    fn = sum(1 for a, b in zip(yt, yp) if a == pos and b != pos)
    n = len(yt)
    p0 = (tp + tn) / n
    pe = (tp + fn) / n * (tp + fp) / n + (fp + tn) / n * (fn + tn) / n
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "pccc": p0,
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "neg_pred_value": tn / (tn + fn) if tn + fn else 0.0,
        "prevalence": (tp + fn) / n,
        "detection_rate": tp / n,
        "balanced_accuracy": 0.5 * ((tp / (tp + fn) if tp + fn else 0.0)
                                    + (tn / (tn + fp) if tn + fp
                                       else 0.0)),
        "kappa": (p0 - pe) / (1.0 - pe) if pe < 1.0 else 0.0,
    }


def matthews_corrcoef(y_true, y_pred, positive=1):
    """eq. (4.13) p.136: MCC = (tp tn - fp fn) /
    sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)); zero when the denominator is."""
    m = binary_metrics(y_true, y_pred, positive)
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(den)


def brier_score(probs, y_true, n_classes=None, halved=False):
    """eq. (4.14) p.136: BS = T^-1 sum_i sum_c (pi_ic - d_ic)^2 with
    d_ic the indicator of the observed category.  The categorical score
    lies in [0, 2]; ``halved`` returns BS/2 in [0, 1] as the book
    suggests."""
    P = _mat(probs)
    yt = [int(v) for v in _flat(y_true)]
    T = len(yt)
    C = n_classes or len(P[0])
    tot = 0.0
    for i, cls in enumerate(yt):
        for c in range(C):
            d = 1.0 if c == cls else 0.0
            tot += (P[i][c] - d) ** 2
    bs = tot / T
    return bs / 2.0 if halved else bs


def mean_log_loss(probs, y_true, n_classes=None):
    """The MLL of p.136: -T^-1 sum_i sum_c 1{y_i = c} log(pi_ic)."""
    P = _mat(probs)
    yt = [int(v) for v in _flat(y_true)]
    T = len(yt)
    tot = 0.0
    for i, cls in enumerate(yt):
        tot += math.log(max(P[i][cls], 1e-300))
    return -tot / T


# ------------------------------- linear mixed models (ch. 5)
def kron(A, B):
    """Kronecker product, the operator used throughout sec. 5.5."""
    A = _mat(A)
    B = _mat(B)
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    out = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    out[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return out


def _logdet(A):
    """log|A| by Gaussian elimination with partial pivoting."""
    M = [row[:] for row in _mat(A)]
    n = len(M)
    ld = 0.0
    sign = 1.0
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-300:
            return float("-inf")
        if piv != i:
            M[i], M[piv] = M[piv], M[i]
            sign = -sign
        ld += math.log(abs(M[i][i]))
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n):
                M[r][c] -= f * M[i][c]
    return ld


def lmm_marginal_v(Z, D, R=None):
    """V = Z D Z' + R, the marginal variance of Y under eq. (5.1)."""
    Z = _mat(Z)
    n = len(Z)
    V = _mm(_mm(Z, _mat(D)), _t(Z))
    if R is None:
        R = [[1.0 if i == j else 0.0 for j in range(n)]
             for i in range(n)]
    R = _mat(R)
    return [[V[i][j] + R[i][j] for j in range(n)] for i in range(n)]


def lmm_loglik(X, Z, y, D, beta=None, R=None):
    """log of eq. (5.2) p.142: L(beta, D, R; y) =
    |V|^{-1/2}(2 pi)^{-n/2} exp(-1/2 (y - X beta)' V^-1 (y - X beta))
    with V = Z D Z' + R.  With ``beta=None`` the GLS estimate is
    plugged in."""
    Xm = _mat(X)
    y = _flat(y)
    n = len(y)
    V = lmm_marginal_v(Z, D, R)
    Vi = _inv(V)
    if beta is None:
        XtVi = _mm(_t(Xm), Vi)
        beta = _solve(_mm(XtVi, Xm), _mv(XtVi, y))
    r = [a - b for a, b in zip(y, _mv(Xm, beta))]
    quad = sum(a * b for a, b in zip(r, _mv(Vi, r)))
    return (-0.5 * n * math.log(2.0 * math.pi) - 0.5 * _logdet(V)
            - 0.5 * quad), beta


def reml_loglik(X, Z, y, D, R=None):
    """The restricted log-likelihood of sec. 5.2.1.2 p.146:
    l_R(theta; y) = -1/2 log|X'V^-1X| - 1/2 log|V|
                    - 1/2 (y - X beta-tilde)' V^-1 (y - X beta-tilde),
    beta-tilde the GLS estimator.  It differs from eq. (5.2) by the
    first term, which is what removes the downward bias of ML."""
    Xm = _mat(X)
    y = _flat(y)
    V = lmm_marginal_v(Z, D, R)
    Vi = _inv(V)
    XtVi = _mm(_t(Xm), Vi)
    A = _mm(XtVi, Xm)
    beta = _solve(A, _mv(XtVi, y))
    r = [a - b for a, b in zip(y, _mv(Xm, beta))]
    quad = sum(a * b for a, b in zip(r, _mv(Vi, r)))
    return (-0.5 * _logdet(A) - 0.5 * _logdet(V) - 0.5 * quad), beta


def em_lmm(X, Z, y, D0=None, sigma2_0=1.0, n_iter=200, tol=1e-10):
    """The EM algorithm of sec. 5.2.1.1 pp.143-144 for R = sigma2 I.

    E step: D-tilde = (D^-1 + sigma^-2 Z'Z)^-1,
            b-tilde  = sigma^-2 D-tilde Z'(y - X beta).
    M step: beta   = (X'X)^-1 X'(y - Z b-tilde),
            sigma2 = n^-1 [tr(Z D-tilde Z')
                           + (y - X beta - Z b-tilde)'(same)],
            D      = D-tilde + b-tilde b-tilde'.
    """
    Xm = _mat(X)
    Zm = _mat(Z)
    y = _flat(y)
    n = len(y)
    q = len(Zm[0])
    D = _mat(D0) if D0 is not None else \
        [[1.0 if i == j else 0.0 for j in range(q)] for i in range(q)]
    s2 = float(sigma2_0)
    XtX = _mm(_t(Xm), Xm)
    ZtZ = _mm(_t(Zm), Zm)
    beta = _solve(XtX, _mv(_t(Xm), y))
    for it in range(int(n_iter)):
        Di = _inv(D)
        A = [[Di[i][j] + ZtZ[i][j] / s2 for j in range(q)]
             for i in range(q)]
        Dt = _inv(A)
        resid = [a - b for a, b in zip(y, _mv(Xm, beta))]
        bt = [v / s2 for v in _mv(Dt, _mv(_t(Zm), resid))]
        beta_new = _solve(XtX, _mv(_t(Xm),
                                   [a - b for a, b in
                                    zip(y, _mv(Zm, bt))]))
        e = [a - b - c for a, b, c in zip(y, _mv(Xm, beta_new),
                                          _mv(Zm, bt))]
        ZDZ = _mm(_mm(Zm, Dt), _t(Zm))
        tr = sum(ZDZ[i][i] for i in range(n))
        s2_new = (tr + sum(v * v for v in e)) / n
        D_new = [[Dt[i][j] + bt[i] * bt[j] for j in range(q)]
                 for i in range(q)]
        gap = max(abs(s2_new - s2),
                  max(abs(a - b) for a, b in zip(beta_new, beta)))
        beta, s2, D = beta_new, s2_new, D_new
        if gap < tol:
            break
    return {"beta": beta, "sigma2": s2, "D": D, "b": bt,
            "iterations": it + 1}


def gblup_model(y, Z_L, G, sigma2_g, sigma2_e=1.0, mu_only=True,
                X=None):
    """eq. (5.3) p.148: Y = 1_n mu + Z_L b + eps with
    b ~ N_J(0, sigma2_g G) and R = sigma2 I_n.  Returns the BLUE of
    the intercept and the BLUP of the genotypic effects."""
    y = _flat(y)
    n = len(y)
    Xm = [[1.0] for _ in range(n)] if (mu_only or X is None) \
        else _mat(X)
    q = len(G)
    Sigma = [[sigma2_g * G[i][j] for j in range(q)] for i in range(q)]
    R = [[sigma2_e if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    beta, b = blue_blup_via_v(Xm, Z_L, y, Sigma, R)
    return {"mu": beta[0], "beta": beta, "b": b}


def gxe_blup_model(y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E,
                   sigma2_e=1.0):
    """eq. (5.4) p.150: Y = 1_n mu + X_E beta_E + Z_L b_1 + Z_EL b_2
    + eps with b_1 ~ N_J(0, sigma2_g G) and b_2 ~ N(0, Sigma_E (x) G),
    Sigma_E the genetic covariance between environments.  The two
    random terms are stacked into one Z = [Z_L Z_EL] and one block
    diagonal Sigma, then solved as eq. (5.1)."""
    y = _flat(y)
    n = len(y)
    Xm = [[1.0] + list(row) for row in _mat(X_E)]
    ZL = _mat(Z_L)
    ZEL = _mat(Z_EL)
    q1 = len(ZL[0])
    S2 = kron(Sigma_E, G)
    q2 = len(S2)
    Z = [ZL[i] + ZEL[i] for i in range(n)]
    Sigma = [[0.0] * (q1 + q2) for _ in range(q1 + q2)]
    for i in range(q1):
        for j in range(q1):
            Sigma[i][j] = sigma2_g * G[i][j]
    for i in range(q2):
        for j in range(q2):
            Sigma[q1 + i][q1 + j] = S2[i][j]
    R = [[sigma2_e if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    beta, b = blue_blup_via_v(Xm, Z, y, Sigma, R)
    return {"beta": beta, "b_lines": b[:q1], "b_gxe": b[q1:]}


def multitrait_model(Y, Z, G, Sigma_T, R_T, X=None):
    """eq. (5.5)/(5.5a) p.153: stacking the n_T traits of each line,
    Y = (1 (x) I_nT) mu + X beta + Z b + eps with b ~ N(0, G (x)
    Sigma_T) and eps ~ N(0, I_J (x) R_nT).  ``Y`` is J x n_T (lines by
    traits); the model is solved in the stacked ordering
    (line 1 traits, line 2 traits, ...).  When Sigma_T and R are
    diagonal this is equivalent to fitting each trait separately
    (book p.153)."""
    Ym = _mat(Y)
    J = len(Ym)
    nT = len(Ym[0])
    y = [v for row in Ym for v in row]
    n = J * nT
    I_nT = [[1.0 if i == j else 0.0 for j in range(nT)]
            for i in range(nT)]
    ones = [[1.0] for _ in range(J)]
    Xm = kron(ones, I_nT)
    if X is not None:
        Xadd = _mat(X)
        Xm = [Xm[i] + Xadd[i] for i in range(n)]
    Zm = kron(Z, I_nT)
    Sigma = kron(G, Sigma_T)
    I_J = [[1.0 if i == j else 0.0 for j in range(J)]
           for i in range(J)]
    R = kron(I_J, R_T)
    beta, b = blue_blup_via_v(Xm, Zm, y, Sigma, R)
    return {"mu": beta[:nT], "beta": beta, "b": b,
            "b_by_line": [b[i * nT:(i + 1) * nT]
                          for i in range(len(b) // nT)]}


def gxe_multitrait_model(Y, Z_L, Z_EL, G, Sigma_T, Sigma_E,
                         Sigma_2T, R_T, I_env, X=None):
    """eq. (5.6) p.155: Y = (1 (x) I_nT) mu + X beta + Z_L b_1
    + Z_EL b_2 + eps with b_1 ~ N(0, G (x) Sigma_T) and
    b_2 ~ N(0, Sigma_E (x) G (x) Sigma_2T).  With Sigma_T, Sigma_2T,
    Sigma_E and R all diagonal the model reduces to separate univariate
    GBLUP fits per trait (book p.155)."""
    Ym = _mat(Y)
    rows = len(Ym)
    nT = len(Ym[0])
    y = [v for row in Ym for v in row]
    n = rows * nT
    I_nT = [[1.0 if i == j else 0.0 for j in range(nT)]
            for i in range(nT)]
    ones = [[1.0] for _ in range(rows)]
    Xm = kron(ones, I_nT)
    if X is not None:
        Xadd = _mat(X)
        Xm = [Xm[i] + Xadd[i] for i in range(n)]
    Z1 = kron(Z_L, I_nT)
    Z2 = kron(Z_EL, I_nT)
    S1 = kron(G, Sigma_T)
    S2 = kron(kron(Sigma_E, G), Sigma_2T)
    q1, q2 = len(S1), len(S2)
    Z = [Z1[i] + Z2[i] for i in range(n)]
    Sigma = [[0.0] * (q1 + q2) for _ in range(q1 + q2)]
    for i in range(q1):
        for j in range(q1):
            Sigma[i][j] = S1[i][j]
    for i in range(q2):
        for j in range(q2):
            Sigma[q1 + i][q1 + j] = S2[i][j]
    I_rows = [[1.0 if i == j else 0.0 for j in range(rows)]
              for i in range(rows)]
    R = kron(I_rows, R_T)
    beta, b = blue_blup_via_v(Xm, Z, y, Sigma, R)
    return {"mu": beta[:nT], "beta": beta,
            "b_lines": b[:q1], "b_gxe": b[q1:]}


# --------------------------- Bayesian genomic regression (ch. 6)
def _rchisq(rng, df):
    """Chi-square draw as a gamma: chi2_df = Ga(df/2, scale 2)."""
    return float(rng.gamma(df / 2.0, 2.0))


def scaled_inv_chisq(rng, nu, S):
    """Scaled inverse chi-square chi^-2(nu, S), the prior used for
    every variance component in ch. 6: a draw is S / chi2_nu."""
    return float(S) / max(_rchisq(rng, float(nu)), 1e-300)


def brr_hyperparameters(y, R2=0.5, nu=5.0, nu_beta=5.0, p=None,
                        sum_var_x=None):
    """The BGLR defaults quoted on pp.175 and 184:
    S = Var(Y)(1 - R2)(nu + 2) and, for the BRR,
    S_beta = Var(Y) R2 (nu_beta + 2); for BayesC the scale is divided
    further by S_x^2 pi_0 with S_x the sum of the column variances of
    X."""
    ys = _flat(y)
    n = len(ys)
    m = sum(ys) / n
    var_y = sum((v - m) ** 2 for v in ys) / (n - 1)
    S = var_y * (1.0 - R2) * (nu + 2.0)
    S_beta = var_y * R2 * (nu_beta + 2.0)
    if sum_var_x:
        S_beta = S_beta / float(sum_var_x)
    return {"S": S, "S_beta": S_beta, "nu": nu, "nu_beta": nu_beta,
            "var_y": var_y}


def bayes_ridge_gibbs(y, X, n_iter=2000, burn_in=500, nu=5.0,
                      nu_beta=5.0, R2=0.5, seed=42):
    """The BRR Gibbs sampler of eq. (6.3), steps 1-6 on pp.174-175:

    sigma2_beta | . ~ chi^-2(nu_beta + p, S_beta + b'b)
    beta        | . ~ N_p(b-tilde, Sigma-tilde),
                  Sigma-tilde = (sigma_beta^-2 I + sigma^-2 X'X)^-1,
                  b-tilde = sigma^-2 Sigma-tilde X'(y - 1 mu)
    mu          | . ~ N(mu-tilde, sigma^2/n),
                  mu-tilde = (1/n) 1'(y - X beta)
    sigma2      | . ~ chi^-2(nu + n, S + ||y - 1 mu - X beta||^2)

    Posterior means over the post-burn-in draws are returned.
    """
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    hp = brr_hyperparameters(ys, R2=R2, nu=nu, nu_beta=nu_beta)
    rng = np.random.default_rng(seed)
    mu = sum(ys) / n
    beta = [0.0] * p
    s2 = hp["S"] / (nu + 2.0)
    s2b = hp["S_beta"] / (nu_beta + 2.0)
    Xt = _t(Xm)
    XtX = _mm(Xt, Xm)
    acc_mu = 0.0
    acc_beta = [0.0] * p
    acc_s2 = 0.0
    acc_s2b = 0.0
    kept = 0
    for it in range(int(n_iter)):
        # sigma2_beta
        s2b = scaled_inv_chisq(rng, nu_beta + p,
                               hp["S_beta"]
                               + sum(b * b for b in beta))
        # beta (joint normal draw via the Cholesky of Sigma-tilde)
        A = [[XtX[i][j] / s2 + ((1.0 / s2b) if i == j else 0.0)
              for j in range(p)] for i in range(p)]
        resid = [a - mu for a in ys]
        rhs = [v / s2 for v in _mv(Xt, resid)]
        mean = _solve(A, rhs)
        Ai = _inv(A)
        L = _chol(Ai)
        z = [float(rng.normal(0, 1)) for _ in range(p)]
        beta = [mean[i] + sum(L[i][k] * z[k] for k in range(p))
                for i in range(p)]
        # mu
        r = [a - b for a, b in zip(ys, _mv(Xm, beta))]
        mu = sum(r) / n + math.sqrt(s2 / n) * float(rng.normal(0, 1))
        # sigma2
        e = [a - mu for a in r]
        s2 = scaled_inv_chisq(rng, nu + n,
                              hp["S"] + sum(v * v for v in e))
        if it >= burn_in:
            kept += 1
            acc_mu += mu
            acc_s2 += s2
            acc_s2b += s2b
            for i in range(p):
                acc_beta[i] += beta[i]
    return {"mu": acc_mu / kept,
            "beta": [v / kept for v in acc_beta],
            "sigma2": acc_s2 / kept, "sigma2_beta": acc_s2b / kept,
            "n_kept": kept, "hyper": hp}


def _chol(A):
    """Lower Cholesky factor; the book uses it on p.177 to turn a
    GBLUP into an equivalent BRR (G = L L')."""
    Am = _mat(A)
    n = len(Am)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = Am[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(s, 1e-300))
            else:
                L[i][j] = s / L[j][j]
    return L


def cholesky_lower(A):
    """Public wrapper around the lower Cholesky factor."""
    return _chol(A)


def bayes_gblup_gibbs(y, G, n_iter=2000, burn_in=500, nu=5.0,
                      nu_g=5.0, R2=0.5, seed=42):
    """Bayesian GBLUP of eq. (6.4) p.176: Y = 1 mu + g + eps with
    g | sigma2_g ~ N(0, sigma2_g G).  The book (p.177) notes this is
    the BRR of eq. (6.3) run on the design X = L, where G = L L' is
    the Cholesky factorization, so that is exactly how it is fitted
    here; the genomic values are g = L beta."""
    L = _chol(G)
    fit = bayes_ridge_gibbs(y, L, n_iter=n_iter, burn_in=burn_in,
                            nu=nu, nu_beta=nu_g, R2=R2, seed=seed)
    fit["g"] = _mv(L, fit["beta"])
    fit["sigma2_g"] = fit.pop("sigma2_beta")
    return fit


def bayes_a_gibbs(y, X, n_iter=2000, burn_in=500, nu=5.0,
                  nu_beta=5.0, R2=0.5, seed=42):
    """BayesA of sec. 6.4 p.178: a separate variance per marker,
    beta_j | sigma2_bj ~ N(0, sigma2_bj) with
    sigma2_bj ~ chi^-2(nu_beta, S_beta).  Step 2.1 of p.178 draws
    sigma2_bj | . ~ chi^-2(nu_beta + 1, S_beta + beta_j^2), which is
    what gives covariate-specific shrinkage."""
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    hp = brr_hyperparameters(ys, R2=R2, nu=nu, nu_beta=nu_beta)
    rng = np.random.default_rng(seed)
    mu = sum(ys) / n
    beta = [0.0] * p
    s2 = hp["S"] / (nu + 2.0)
    s2j = [hp["S_beta"] / (nu_beta + 2.0)] * p
    Xt = _t(Xm)
    col_ss = [sum(v * v for v in col) for col in Xt]
    acc_mu = acc_s2 = 0.0
    acc_beta = [0.0] * p
    acc_s2j = [0.0] * p
    kept = 0
    for it in range(int(n_iter)):
        for j in range(p):
            s2j[j] = scaled_inv_chisq(rng, nu_beta + 1.0,
                                      hp["S_beta"] + beta[j] ** 2)
        # single-site updates of beta (p.181 form)
        for j in range(p):
            partial = [ys[i] - mu - sum(Xm[i][k] * beta[k]
                                        for k in range(p) if k != j)
                       for i in range(n)]
            prec = 1.0 / s2j[j] + col_ss[j] / s2
            var = 1.0 / prec
            mean = var * sum(Xm[i][j] * partial[i]
                             for i in range(n)) / s2
            beta[j] = mean + math.sqrt(var) * float(rng.normal(0, 1))
        r = [a - b for a, b in zip(ys, _mv(Xm, beta))]
        mu = sum(r) / n + math.sqrt(s2 / n) * float(rng.normal(0, 1))
        e = [a - mu for a in r]
        s2 = scaled_inv_chisq(rng, nu + n,
                              hp["S"] + sum(v * v for v in e))
        if it >= burn_in:
            kept += 1
            acc_mu += mu
            acc_s2 += s2
            for j in range(p):
                acc_beta[j] += beta[j]
                acc_s2j[j] += s2j[j]
    return {"mu": acc_mu / kept,
            "beta": [v / kept for v in acc_beta],
            "sigma2": acc_s2 / kept,
            "sigma2_beta_j": [v / kept for v in acc_s2j],
            "n_kept": kept, "hyper": hp}


def bayes_c_gibbs(y, X, n_iter=2000, burn_in=500, nu=5.0,
                  nu_beta=5.0, R2=0.5, pi0=0.5, phi0=10.0, seed=42):
    """BayesC of sec. 6.5 pp.180-183: beta_j is a spike-and-slab,
    pi N(0, sigma2_beta) + (1 - pi) DG(0), with a latent Bernoulli
    Z_j.  The joint (beta_j, Z_j) update of p.182 is used, drawing
    Z_j ~ Ber(pi-tilde) with
    pi-tilde = pi sqrt(sigma-tilde_j^2/sigma2_beta) /
               (pi sqrt(...) + 1 - pi)
    and then beta_j ~ N(b-tilde_j, sigma-tilde_j^2) when Z_j = 1,
    beta_j = 0 otherwise.  pi | . ~ Beta(phi0 pi0 + p z-bar,
    phi0(1 - pi0) + p(1 - z-bar)) (p.183).  With pi0 = 1 the model
    reduces to the BRR (p.180)."""
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    hp = brr_hyperparameters(ys, R2=R2, nu=nu, nu_beta=nu_beta)
    rng = np.random.default_rng(seed)
    mu = sum(ys) / n
    beta = [0.0] * p
    z = [1] * p
    pi = float(pi0)
    s2 = hp["S"] / (nu + 2.0)
    s2b = hp["S_beta"] / (nu_beta + 2.0)
    Xt = _t(Xm)
    col_ss = [sum(v * v for v in col) for col in Xt]
    acc_mu = acc_s2 = acc_s2b = acc_pi = 0.0
    acc_beta = [0.0] * p
    acc_z = [0.0] * p
    kept = 0
    for it in range(int(n_iter)):
        for j in range(p):
            partial = [ys[i] - mu - sum(Xm[i][k] * beta[k]
                                        for k in range(p) if k != j)
                       for i in range(n)]
            prec = 1.0 / s2b + col_ss[j] / s2
            var = 1.0 / prec
            mean = var * sum(Xm[i][j] * partial[i]
                             for i in range(n)) / s2
            ratio = math.sqrt(var / s2b) \
                * math.exp(min(mean * mean / (2.0 * var), 700.0))
            pt = pi * ratio / (pi * ratio + (1.0 - pi)) \
                if pi < 1.0 else 1.0
            z[j] = 1 if float(rng.uniform(0, 1)) < pt else 0
            beta[j] = (mean + math.sqrt(var)
                       * float(rng.normal(0, 1))) if z[j] else 0.0
        zbar = sum(z) / p
        s2b = scaled_inv_chisq(
            rng, nu_beta + p * zbar,
            hp["S_beta"] + sum(zj * b * b for zj, b in zip(z, beta)))
        if pi0 < 1.0:
            a = phi0 * pi0 + p * zbar
            b_ = phi0 * (1.0 - pi0) + p * (1.0 - zbar)
            g1 = float(rng.gamma(max(a, 1e-6), 1.0))
            g2 = float(rng.gamma(max(b_, 1e-6), 1.0))
            pi = g1 / (g1 + g2)
        r = [a - b for a, b in zip(ys, _mv(Xm, beta))]
        mu = sum(r) / n + math.sqrt(s2 / n) * float(rng.normal(0, 1))
        e = [a - mu for a in r]
        s2 = scaled_inv_chisq(rng, nu + n,
                              hp["S"] + sum(v * v for v in e))
        if it >= burn_in:
            kept += 1
            acc_mu += mu
            acc_s2 += s2
            acc_s2b += s2b
            acc_pi += pi
            for j in range(p):
                acc_beta[j] += beta[j]
                acc_z[j] += z[j]
    return {"mu": acc_mu / kept,
            "beta": [v / kept for v in acc_beta],
            "sigma2": acc_s2 / kept, "sigma2_beta": acc_s2b / kept,
            "pi": acc_pi / kept,
            "inclusion_prob": [v / kept for v in acc_z],
            "n_kept": kept, "hyper": hp}


def bayes_b_gibbs(y, X, n_iter=2000, burn_in=500, pi0=0.5,
                  phi0=10.0, seed=42, **kw):
    """BayesB of p.183: BayesA with a mixture prior, i.e. the
    spike-and-slab of BayesC combined with a marker-specific slab
    variance.  With pi0 = 1 it reduces to BayesA (p.183)."""
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    nu = kw.get("nu", 5.0)
    nu_beta = kw.get("nu_beta", 5.0)
    hp = brr_hyperparameters(ys, R2=kw.get("R2", 0.5), nu=nu,
                             nu_beta=nu_beta)
    rng = np.random.default_rng(seed)
    mu = sum(ys) / n
    beta = [0.0] * p
    z = [1] * p
    pi = float(pi0)
    s2 = hp["S"] / (nu + 2.0)
    s2j = [hp["S_beta"] / (nu_beta + 2.0)] * p
    Xt = _t(Xm)
    col_ss = [sum(v * v for v in col) for col in Xt]
    acc_mu = acc_s2 = acc_pi = 0.0
    acc_beta = [0.0] * p
    acc_z = [0.0] * p
    kept = 0
    for it in range(int(n_iter)):
        for j in range(p):
            s2j[j] = scaled_inv_chisq(rng, nu_beta + 1.0,
                                      hp["S_beta"] + beta[j] ** 2)
            partial = [ys[i] - mu - sum(Xm[i][k] * beta[k]
                                        for k in range(p) if k != j)
                       for i in range(n)]
            prec = 1.0 / s2j[j] + col_ss[j] / s2
            var = 1.0 / prec
            mean = var * sum(Xm[i][j] * partial[i]
                             for i in range(n)) / s2
            ratio = math.sqrt(var / s2j[j]) \
                * math.exp(min(mean * mean / (2.0 * var), 700.0))
            pt = pi * ratio / (pi * ratio + (1.0 - pi)) \
                if pi < 1.0 else 1.0
            z[j] = 1 if float(rng.uniform(0, 1)) < pt else 0
            beta[j] = (mean + math.sqrt(var)
                       * float(rng.normal(0, 1))) if z[j] else 0.0
        if pi0 < 1.0:
            zbar = sum(z) / p
            a = phi0 * pi0 + p * zbar
            b_ = phi0 * (1.0 - pi0) + p * (1.0 - zbar)
            g1 = float(rng.gamma(max(a, 1e-6), 1.0))
            g2 = float(rng.gamma(max(b_, 1e-6), 1.0))
            pi = g1 / (g1 + g2)
        r = [a - b for a, b in zip(ys, _mv(Xm, beta))]
        mu = sum(r) / n + math.sqrt(s2 / n) * float(rng.normal(0, 1))
        e = [a - mu for a in r]
        s2 = scaled_inv_chisq(rng, nu + n,
                              hp["S"] + sum(v * v for v in e))
        if it >= burn_in:
            kept += 1
            acc_mu += mu
            acc_s2 += s2
            acc_pi += pi
            for j in range(p):
                acc_beta[j] += beta[j]
                acc_z[j] += z[j]
    return {"mu": acc_mu / kept,
            "beta": [v / kept for v in acc_beta],
            "sigma2": acc_s2 / kept, "pi": acc_pi / kept,
            "inclusion_prob": [v / kept for v in acc_z],
            "n_kept": kept, "hyper": hp}


def bayes_lasso_gibbs(y, X, n_iter=2000, burn_in=500, nu=5.0,
                      R2=0.5, lam2=1.0, seed=42):
    """The Bayesian Lasso of sec. 6.6 p.184 in the Park and Casella
    (2008) scale-mixture form quoted there:
    beta_j | tau_j ~ N(0, tau_j sigma2) with tau_j ~ Exp(2/lambda^2),
    which is the double exponential L(0, sqrt(sigma2)/lambda) after
    marginalizing tau.  The heavier tails shrink small effects harder
    and large effects less than the BRR."""
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    hp = brr_hyperparameters(ys, R2=R2, nu=nu)
    rng = np.random.default_rng(seed)
    mu = sum(ys) / n
    beta = [0.0] * p
    tau = [1.0] * p
    s2 = hp["S"] / (nu + 2.0)
    Xt = _t(Xm)
    col_ss = [sum(v * v for v in col) for col in Xt]
    acc_mu = acc_s2 = 0.0
    acc_beta = [0.0] * p
    acc_tau = [0.0] * p
    kept = 0
    for it in range(int(n_iter)):
        for j in range(p):
            # tau_j^-1 | . is inverse Gaussian; sample tau_j through
            # its gamma-mixture representation, which keeps the chain
            # in closed form without an extra sampler
            b2 = max(beta[j] ** 2, 1e-12)
            shape = 0.5
            scale = 2.0 / (lam2 + b2 / max(s2, 1e-300))
            tau[j] = max(float(rng.gamma(shape, scale)), 1e-12)
            partial = [ys[i] - mu - sum(Xm[i][k] * beta[k]
                                        for k in range(p) if k != j)
                       for i in range(n)]
            prec = 1.0 / (tau[j] * s2) + col_ss[j] / s2
            var = 1.0 / prec
            mean = var * sum(Xm[i][j] * partial[i]
                             for i in range(n)) / s2
            beta[j] = mean + math.sqrt(var) * float(rng.normal(0, 1))
        r = [a - b for a, b in zip(ys, _mv(Xm, beta))]
        mu = sum(r) / n + math.sqrt(s2 / n) * float(rng.normal(0, 1))
        e = [a - mu for a in r]
        s2 = scaled_inv_chisq(rng, nu + n,
                              hp["S"] + sum(v * v for v in e))
        if it >= burn_in:
            kept += 1
            acc_mu += mu
            acc_s2 += s2
            for j in range(p):
                acc_beta[j] += beta[j]
                acc_tau[j] += tau[j]
    return {"mu": acc_mu / kept,
            "beta": [v / kept for v in acc_beta],
            "sigma2": acc_s2 / kept,
            "tau": [v / kept for v in acc_tau],
            "n_kept": kept, "hyper": hp}


def extended_predictor(n, X_E=None, X=None, X_EM=None):
    """eq. (6.6) p.186: y = 1_n mu + X_E beta_E + X beta
    + X_EM beta_EM + eps.  Assembles the stacked design matrix in that
    order and reports the block widths, so each block can be given its
    own prior (FIXED, BRR, BayesA/B/C, BL) as the book describes."""
    blocks = [("intercept", [[1.0] for _ in range(n)])]
    for name, B in (("environments", X_E), ("markers", X),
                    ("env_x_marker", X_EM)):
        if B is not None:
            blocks.append((name, _mat(B)))
    design = [sum((blk[i] for _, blk in blocks), []) for i in range(n)]
    widths = {name: len(blk[0]) for name, blk in blocks}
    return {"design": design, "widths": widths,
            "n_columns": len(design[0])}


def rkhs_covariances(Z_L, G, Z_LE=None, I_env=None, sigma2_g=1.0,
                     sigma2_ge=1.0):
    """eq. (6.7) p.186: under the RKHS parameterization the predictor
    terms Z_L g and Z_LE gE enter through their covariance matrices
    K_L = Z_L G Z_L' and K_LE = Z_LE (I (x) G) Z_LE', which is the
    precalculation the book says BGLR needs (the same trick as
    K_L = Z G Z' for eq. 6.5 on p.177)."""
    ZL = _mat(Z_L)
    K_L = _mm(_mm(ZL, _mat(G)), _t(ZL))
    K_L = [[sigma2_g * v for v in row] for row in K_L]
    out = {"K_L": K_L}
    if Z_LE is not None and I_env is not None:
        ZLE = _mat(Z_LE)
        IG = kron(I_env, G)
        K_LE = _mm(_mm(ZLE, IG), _t(ZLE))
        out["K_LE"] = [[sigma2_ge * v for v in row] for row in K_LE]
    return out


# ---------------- Bayesian multi-trait / BMTME (ch. 6 pp.190-196)
def inv_wishart_draw(rng, nu, S):
    """Inverse-Wishart IW(nu, S) via the Bartlett decomposition of a
    Wishart draw: if W ~ Wishart(nu, S^-1) then W^-1 ~ IW(nu, S).
    Used for Sigma_T, Sigma_E and R in eq. (6.8)-(6.11)."""
    Sm = _mat(S)
    p = len(Sm)
    Sinv = _inv(Sm)
    L = _chol(Sinv)
    A = [[0.0] * p for _ in range(p)]
    for i in range(p):
        A[i][i] = math.sqrt(max(_rchisq(rng, nu - i), 1e-300))
        for j in range(i):
            A[i][j] = float(rng.normal(0, 1))
    LA = _mm(L, A)
    W = _mm(LA, _t(LA))
    return _inv(W)


def multitrait_bayes_gibbs(Y, Z1, G, X=None, n_iter=1500,
                           burn_in=400, nu_T=None, S_T=None,
                           nu_R=None, S_R=None, seed=42):
    """The Gibbs sampler of eq. (6.9), steps 1-6 on p.193.

    Y = 1_J mu' + X B + Z_1 b_1 + E with
    E ~ MN(0, I_J, R) and b_1 ~ MN(0, G, Sigma_T):

      mu      ~ N(mu-tilde, R/J),  mu-tilde = colmean(Y - XB - Z1 b1)
      g       ~ N(g-tilde, G-tilde),
                G-tilde = [(Sigma_T^-1 (x) G^-1)
                           + (R^-1 (x) Z1'Z1)]^-1,
                g-tilde = G-tilde (R^-1 (x) Z1') vec(Y - 1 mu' - XB)
      Sigma_T ~ IW(nu_T + J, b1' G^-1 b1 + S_T)
      R       ~ IW(nu_R + J, S_R + resid' resid)

    ``vec`` stacks columns, matching the book's convention.  With
    Sigma_T and R diagonal the fit reduces to a univariate GBLUP per
    trait (p.191).
    """
    Ym = _mat(Y)
    J = len(Ym)
    nT = len(Ym[0])
    Z = _mat(Z1)
    q = len(Z[0])
    Gm = _mat(G)
    Ginv = _inv(Gm)
    rng = np.random.default_rng(seed)
    nu_T = float(nu_T if nu_T is not None else nT + 2)
    nu_R = float(nu_R if nu_R is not None else nT + 2)
    S_T = _mat(S_T) if S_T is not None else \
        [[1.0 if i == j else 0.0 for j in range(nT)]
         for i in range(nT)]
    S_R = _mat(S_R) if S_R is not None else \
        [[1.0 if i == j else 0.0 for j in range(nT)]
         for i in range(nT)]
    mu = [sum(row[t] for row in Ym) / J for t in range(nT)]
    b1 = [[0.0] * nT for _ in range(q)]
    Sig_T = [[1.0 if i == j else 0.0 for j in range(nT)]
             for i in range(nT)]
    R = [[1.0 if i == j else 0.0 for j in range(nT)]
         for i in range(nT)]
    ZtZ = _mm(_t(Z), Z)
    acc_mu = [0.0] * nT
    acc_b1 = [[0.0] * nT for _ in range(q)]
    acc_ST = [[0.0] * nT for _ in range(nT)]
    acc_R = [[0.0] * nT for _ in range(nT)]
    kept = 0
    for it in range(int(n_iter)):
        Rinv = _inv(R)
        STinv = _inv(Sig_T)
        # 2. mu
        resid = [[Ym[i][t] - sum(Z[i][k] * b1[k][t]
                                 for k in range(q))
                  for t in range(nT)] for i in range(J)]
        cm = [sum(row[t] for row in resid) / J for t in range(nT)]
        Lmu = _chol([[R[i][j] / J for j in range(nT)]
                     for i in range(nT)])
        z = [float(rng.normal(0, 1)) for _ in range(nT)]
        mu = [cm[t] + sum(Lmu[t][k] * z[k] for k in range(nT))
              for t in range(nT)]
        # 3. g = vec(b1): (Sigma_T^-1 (x) G^-1) + (R^-1 (x) Z1'Z1)
        A = [[0.0] * (q * nT) for _ in range(q * nT)]
        for s in range(nT):
            for t in range(nT):
                for a in range(q):
                    for b in range(q):
                        A[s * q + a][t * q + b] = \
                            STinv[s][t] * Ginv[a][b] \
                            + Rinv[s][t] * ZtZ[a][b]
        Ycen = [[Ym[i][t] - mu[t] for t in range(nT)]
                for i in range(J)]
        rhs = [0.0] * (q * nT)
        ZtY = _mm(_t(Z), Ycen)
        for s in range(nT):
            for a in range(q):
                rhs[s * q + a] = sum(Rinv[s][t] * ZtY[a][t]
                                     for t in range(nT))
        mean = _solve(A, rhs)
        Ai = _inv(A)
        L = _chol(Ai)
        zz = [float(rng.normal(0, 1)) for _ in range(q * nT)]
        g = [mean[i] + sum(L[i][k] * zz[k] for k in range(q * nT))
             for i in range(q * nT)]
        b1 = [[g[t * q + a] for t in range(nT)] for a in range(q)]
        # 4. Sigma_T ~ IW(nu_T + J, b1' G^-1 b1 + S_T)
        GB = _mm(Ginv, b1)
        BtGB = _mm(_t(b1), GB)
        Sig_T = inv_wishart_draw(
            rng, nu_T + q,
            [[BtGB[i][j] + S_T[i][j] for j in range(nT)]
             for i in range(nT)])
        # 5. R ~ IW(nu_R + J, S_R + E'E)
        E = [[Ycen[i][t] - sum(Z[i][k] * b1[k][t]
                               for k in range(q))
              for t in range(nT)] for i in range(J)]
        EtE = _mm(_t(E), E)
        R = inv_wishart_draw(
            rng, nu_R + J,
            [[EtE[i][j] + S_R[i][j] for j in range(nT)]
             for i in range(nT)])
        if it >= burn_in:
            kept += 1
            for t in range(nT):
                acc_mu[t] += mu[t]
                for s in range(nT):
                    acc_ST[t][s] += Sig_T[t][s]
                    acc_R[t][s] += R[t][s]
            for a in range(q):
                for t in range(nT):
                    acc_b1[a][t] += b1[a][t]
    return {"mu": [v / kept for v in acc_mu],
            "b1": [[v / kept for v in row] for row in acc_b1],
            "Sigma_T": [[v / kept for v in row] for row in acc_ST],
            "R": [[v / kept for v in row] for row in acc_R],
            "n_kept": kept}


def multitrait_ridge_form(Z1, G):
    """eq. (6.10) p.194: the multivariate ridge form of eq. (6.9),
    X_1 = Z_1 L_G with G = L_G L_G' the Cholesky factorization and
    B_1 = L_G^-1 b_1 ~ MN(0, I_J, Sigma_T).  Returns X_1 and L_G, so
    a BRR-style predictor can replace the RKHS one."""
    L = _chol(_mat(G))
    X1 = _mm(_mat(Z1), L)
    return {"X1": X1, "L_G": L}


def bmtme_conditionals(Y, Z1, Z2, G, Sigma_T, Sigma_E, R, mu=None,
                       b1=None, b2=None, nu_T=None, S_T=None,
                       nu_E=None, S_E=None):
    """eq. (6.11) p.195 and its Gibbs steps p.196 (BMTME).

    Y = 1_IJ mu' + X B + Z_1 b_1 + Z_2 b_2 + E with
    b_1 ~ MN(0, G, Sigma_T) and b_2 ~ MN(0, Sigma_E (x) G, Sigma_T).
    Returns the two inverse-Wishart scale matrices and degrees of
    freedom of steps 5 and 6:

      Sigma_T | . ~ IW(nu_T + J + IJ,
                       b1'G^-1 b1 + b2'(Sigma_E^-1 (x) G^-1)b2 + S_T)
      Sigma_E | . ~ IW(nu_E + J L,
                       b2*'(G^-1 (x) Sigma_T^-1) b2* + S_E)
    """
    Ym = _mat(Y)
    nT = len(Ym[0])
    Gm = _mat(G)
    Ginv = _inv(Gm)
    J = len(Gm)
    b1 = _mat(b1) if b1 is not None else \
        [[0.0] * nT for _ in range(J)]
    q2 = len(_mat(Z2)[0])
    b2 = _mat(b2) if b2 is not None else \
        [[0.0] * nT for _ in range(q2)]
    SEinv = _inv(_mat(Sigma_E))
    I = len(SEinv)
    nu_T = float(nu_T if nu_T is not None else nT + 2)
    nu_E = float(nu_E if nu_E is not None else I + 2)
    S_T = _mat(S_T) if S_T is not None else \
        [[1.0 if i == j else 0.0 for j in range(nT)]
         for i in range(nT)]
    S_E = _mat(S_E) if S_E is not None else \
        [[1.0 if i == j else 0.0 for j in range(I)]
         for i in range(I)]
    # b1' G^-1 b1
    term1 = _mm(_t(b1), _mm(Ginv, b1))
    # b2' (Sigma_E^-1 (x) G^-1) b2
    K = kron(SEinv, Ginv)
    term2 = _mm(_t(b2), _mm(K, b2))
    scale_T = [[term1[i][j] + term2[i][j] + S_T[i][j]
                for j in range(nT)] for i in range(nT)]
    # Sigma_E scale uses b2 reshaped so that vec(b2*') = vec(b2')
    STinv = _inv(_mat(Sigma_T))
    b2s = [[b2[e * J + a][t] if e * J + a < len(b2) else 0.0
            for t in range(nT) for a in range(J)]
           for e in range(I)]
    KE = kron(Ginv, STinv)
    inner = _mm(b2s, _mm(KE, _t(b2s)))
    scale_E = [[inner[i][j] + S_E[i][j] for j in range(I)]
               for i in range(I)]
    return {"nu_T_post": nu_T + J + len(b2),
            "scale_T": scale_T,
            "nu_E_post": nu_E + J * I,
            "scale_E": scale_E}
