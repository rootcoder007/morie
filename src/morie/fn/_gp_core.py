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
    """Solve A x = b through the native linear-algebra core."""
    x = np.linalg.solve(np.marr(_mat(A)), np.marr(_flat(b)))
    return [float(v) for v in x._flat()]


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
