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
    # An eigenvector is defined only up to sign, and eigh and R's eigen
    # need not pick the same one. Pin it: each column's
    # largest-magnitude entry is made positive, the convention
    # _tail1core.eigsym already uses. Without this the eigenvalues agree
    # and the loadings silently differ by -1.
    for j in range(len(W[0])):
        col = [W[r][j] for r in range(len(W))]
        piv = max(range(len(col)), key=lambda r: abs(col[r]))
        if col[piv] < 0.0:
            for r in range(len(W)):
                W[r][j] = -W[r][j]
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


# ------------- ordinal / categorical / count models (ch. 7)
def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(u):
    """Inverse standard normal CDF (Acklam's rational approximation,
    refined by one Halley step against ``_norm_cdf``)."""
    if u <= 0.0:
        return -40.0
    if u >= 1.0:
        return 40.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1.0 - 0.02425
    if u < pl:
        q = math.sqrt(-2.0 * math.log(u))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q
              + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif u > ph:
        q = math.sqrt(-2.0 * math.log(1.0 - u))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q
               + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = u - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r
              + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
              + b[4]) * r + 1.0)
    e = _norm_cdf(x) - u
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if pdf > 0:
        x = x - e / pdf
    return x


def ordinal_probabilities(eta, thresholds, link="probit"):
    """eq. (7.1) p.210: p_ic = F(gamma_c + x_i'beta)
    - F(gamma_{c-1} + x_i'beta), c = 1..C, with gamma_0 = -inf and
    gamma_C = +inf.  ``link`` selects the standard normal CDF (the
    ordinal probit model) or the standard logistic CDF (the ordinal
    logistic model), both given on p.210."""
    g = _flat(thresholds)
    F = _norm_cdf if link == "probit" else \
        (lambda z: 1.0 / (1.0 + math.exp(-z)))
    out = []
    for e in _flat(eta):
        cuts = [0.0] + [F(gc + e) for gc in g] + [1.0]
        out.append([cuts[c + 1] - cuts[c]
                    for c in range(len(g) + 1)])
    return out


def _rtruncnorm(rng, mean, sd, lo, hi):
    """Draw from N(mean, sd^2) truncated to (lo, hi) by inverting the
    CDF, which is the step the Albert and Chib (1993) sampler of
    p.212 needs for the latent variables."""
    a = _norm_cdf((lo - mean) / sd) if lo > -1e300 else 0.0
    b = _norm_cdf((hi - mean) / sd) if hi < 1e300 else 1.0
    if b <= a:
        return mean
    u = a + (b - a) * float(rng.uniform(0, 1))
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return mean + sd * _norm_ppf(u)


def ordinal_probit_gibbs(y, X, n_iter=1500, burn_in=400,
                         nu_beta=5.0, S_beta=None, seed=42):
    """The Gibbs sampler for the Bayesian ordinal probit model of
    eq. (7.1), steps 1-6 on pp.212-213 (Albert and Chib 1993):

      l_i     | . ~ N(-x_i'beta, 1) truncated to
                    (gamma_{y_i - 1}, gamma_{y_i})
      beta_j  | . ~ N(b-tilde_j, s-tilde_j^2),
                    s-tilde_j^2 = (sigma_beta^-2 + x_j'x_j)^-1,
                    b-tilde_j = -s-tilde_j^2 (x_j' e_j)
      gamma_c | . ~ U(a_c, b_c),
                    a_c = max{l_i : y_i = c},
                    b_c = min{l_i : y_i = c + 1}
      sigma2_beta | . ~ chi^-2(nu_beta + p, S_beta + beta'beta)

    ``y`` holds category labels 1..C.
    """
    ys = [int(v) for v in _flat(y)]
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    C = max(ys)
    rng = np.random.default_rng(seed)
    if S_beta is None:
        S_beta = 1.0
    beta = [0.0] * p
    s2b = 1.0
    gamma = [float(c) - C / 2.0 for c in range(1, C)]
    l = [0.0] * n
    Xt = _t(Xm)
    col_ss = [sum(v * v for v in col) for col in Xt]
    acc_beta = [0.0] * p
    acc_gamma = [0.0] * max(len(gamma), 1)
    acc_s2b = 0.0
    kept = 0
    for it in range(int(n_iter)):
        eta = _mv(Xm, beta)
        for i in range(n):
            c = ys[i]
            lo = gamma[c - 2] if c >= 2 else -1e300
            hi = gamma[c - 1] if c <= C - 1 else 1e300
            l[i] = _rtruncnorm(rng, -eta[i], 1.0, lo, hi)
        for j in range(p):
            e_j = [l[i] + sum(Xm[i][k] * beta[k]
                              for k in range(p) if k != j)
                   for i in range(n)]
            var = 1.0 / (1.0 / s2b + col_ss[j])
            mean = -var * sum(Xm[i][j] * e_j[i] for i in range(n))
            beta[j] = mean + math.sqrt(var) * float(rng.normal(0, 1))
        for c in range(1, C):
            a_c = max([l[i] for i in range(n) if ys[i] == c],
                      default=-1e300)
            b_c = min([l[i] for i in range(n) if ys[i] == c + 1],
                      default=1e300)
            lo = max(a_c, gamma[c - 2] if c >= 2 else -1e300)
            hi = min(b_c, gamma[c] if c <= C - 2 else 1e300)
            if hi > lo:
                gamma[c - 1] = lo + (hi - lo) \
                    * float(rng.uniform(0, 1))
        s2b = scaled_inv_chisq(rng, nu_beta + p,
                               S_beta + sum(b * b for b in beta))
        if it >= burn_in:
            kept += 1
            acc_s2b += s2b
            for j in range(p):
                acc_beta[j] += beta[j]
            for c in range(len(gamma)):
                acc_gamma[c] += gamma[c]
    return {"beta": [v / kept for v in acc_beta],
            "gamma": [v / kept for v in acc_gamma[:len(gamma)]],
            "sigma2_beta": acc_s2b / kept, "n_kept": kept,
            "n_categories": C}


def ordinal_probit_gblup_gibbs(y, G, n_iter=1500, burn_in=400,
                               nu_g=5.0, S_g=None, seed=42):
    """The ordinal probit GBLUP of eq. (7.2) p.214:
    p_ic = Phi(gamma_c + b_i) - Phi(gamma_{c-1} + b_i) with
    b | sigma2_g ~ N(0, sigma2_g G).  Gibbs steps 1-6 on p.214:

      b        | . ~ N(b-tilde, Sigma-tilde_b),
                     Sigma-tilde_b = (sigma_g^-2 G^-1 + I_n)^-1,
                     b-tilde = -Sigma-tilde_b l
      sigma2_g | . ~ chi^-2(nu_g + n, S_g + b' G^-1 b)

    with the same latent-variable and threshold steps as eq. (7.1).
    """
    ys = [int(v) for v in _flat(y)]
    n = len(ys)
    C = max(ys)
    Gm = _mat(G)
    Ginv = _inv(Gm)
    rng = np.random.default_rng(seed)
    if S_g is None:
        S_g = 1.0
    b = [0.0] * n
    s2g = 1.0
    gamma = [float(c) - C / 2.0 for c in range(1, C)]
    l = [0.0] * n
    acc_b = [0.0] * n
    acc_gamma = [0.0] * max(len(gamma), 1)
    acc_s2g = 0.0
    kept = 0
    for it in range(int(n_iter)):
        for i in range(n):
            c = ys[i]
            lo = gamma[c - 2] if c >= 2 else -1e300
            hi = gamma[c - 1] if c <= C - 1 else 1e300
            l[i] = _rtruncnorm(rng, -b[i], 1.0, lo, hi)
        A = [[Ginv[i][j] / s2g + (1.0 if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        Sig = _inv(A)
        mean = [-v for v in _mv(Sig, l)]
        L = _chol(Sig)
        z = [float(rng.normal(0, 1)) for _ in range(n)]
        b = [mean[i] + sum(L[i][k] * z[k] for k in range(n))
             for i in range(n)]
        for c in range(1, C):
            a_c = max([l[i] for i in range(n) if ys[i] == c],
                      default=-1e300)
            b_c = min([l[i] for i in range(n) if ys[i] == c + 1],
                      default=1e300)
            lo = max(a_c, gamma[c - 2] if c >= 2 else -1e300)
            hi = min(b_c, gamma[c] if c <= C - 2 else 1e300)
            if hi > lo:
                gamma[c - 1] = lo + (hi - lo) \
                    * float(rng.uniform(0, 1))
        quad = sum(b[i] * sum(Ginv[i][j] * b[j] for j in range(n))
                   for i in range(n))
        s2g = scaled_inv_chisq(rng, nu_g + n, S_g + quad)
        if it >= burn_in:
            kept += 1
            acc_s2g += s2g
            for i in range(n):
                acc_b[i] += b[i]
            for c in range(len(gamma)):
                acc_gamma[c] += gamma[c]
    return {"b": [v / kept for v in acc_b],
            "gamma": [v / kept for v in acc_gamma[:len(gamma)]],
            "sigma2_g": acc_s2g / kept, "n_kept": kept,
            "n_categories": C}


# ------- ordinal logistic / multinomial / Poisson (ch. 7 pp.221-233)
def _rpolya_gamma(rng, b, c, n_terms=120):
    """PG(b, c) by the Devroye-style infinite convolution used in
    Polson, Scott and Windle (2013), the augmentation the book adopts
    on p.222: omega = (2 pi^2)^-1 sum_k g_k / ((k - 1/2)^2
    + c^2/(4 pi^2)) with g_k ~ Ga(b, 1)."""
    tot = 0.0
    cc = (c * c) / (4.0 * math.pi * math.pi)
    for k in range(1, int(n_terms) + 1):
        g = float(rng.gamma(b, 1.0))
        tot += g / ((k - 0.5) ** 2 + cc)
    return tot / (2.0 * math.pi * math.pi)


def ordinal_logistic_gibbs(y, X, n_iter=800, burn_in=200,
                           nu_beta=5.0, S_beta=1.0, seed=42):
    """The ordinal logistic Gibbs sampler of sec. 7.3, steps 1-7 on
    p.224, using the Polya-Gamma augmentation of p.222:

      omega_i | . ~ PG(2, l_i + eta_i)
      l_i     | . ~ N(-x_i'beta, omega_i^-1) truncated to
                    (gamma_{y_i - 1}, gamma_{y_i})
      beta_j  | . ~ N(b-tilde_j, s-tilde_j^2),
                    s-tilde_j^2 = (sigma_beta^-2
                                   + sum_i omega_i x_ij^2)^-1,
                    b-tilde_j = -s-tilde_j^2 sum_i omega_i x_ij e_ij
      gamma_c | . ~ U(a_c, b_c)
      sigma2_beta | . ~ chi^-2(nu_beta + p, S_beta + beta'beta)
    """
    ys = [int(v) for v in _flat(y)]
    Xm = _mat(X)
    n = len(ys)
    p = len(Xm[0])
    C = max(ys)
    rng = np.random.default_rng(seed)
    beta = [0.0] * p
    s2b = 1.0
    gamma = [float(c) - C / 2.0 for c in range(1, C)]
    l = [0.0] * n
    omega = [1.0] * n
    acc_beta = [0.0] * p
    acc_gamma = [0.0] * max(len(gamma), 1)
    kept = 0
    for it in range(int(n_iter)):
        eta = _mv(Xm, beta)
        for i in range(n):
            omega[i] = max(_rpolya_gamma(rng, 2.0, l[i] + eta[i]),
                           1e-9)
        for i in range(n):
            c = ys[i]
            lo = gamma[c - 2] if c >= 2 else -1e300
            hi = gamma[c - 1] if c <= C - 1 else 1e300
            l[i] = _rtruncnorm(rng, -eta[i],
                               1.0 / math.sqrt(omega[i]), lo, hi)
        for j in range(p):
            e_j = [l[i] + sum(Xm[i][k] * beta[k]
                              for k in range(p) if k != j)
                   for i in range(n)]
            prec = 1.0 / s2b + sum(omega[i] * Xm[i][j] ** 2
                                   for i in range(n))
            var = 1.0 / prec
            mean = -var * sum(omega[i] * Xm[i][j] * e_j[i]
                              for i in range(n))
            beta[j] = mean + math.sqrt(var) * float(rng.normal(0, 1))
        for c in range(1, C):
            a_c = max([l[i] for i in range(n) if ys[i] == c],
                      default=-1e300)
            b_c = min([l[i] for i in range(n) if ys[i] == c + 1],
                      default=1e300)
            lo = max(a_c, gamma[c - 2] if c >= 2 else -1e300)
            hi = min(b_c, gamma[c] if c <= C - 2 else 1e300)
            if hi > lo:
                gamma[c - 1] = lo + (hi - lo) \
                    * float(rng.uniform(0, 1))
        s2b = scaled_inv_chisq(rng, nu_beta + p,
                               S_beta + sum(b * b for b in beta))
        if it >= burn_in:
            kept += 1
            for j in range(p):
                acc_beta[j] += beta[j]
            for c in range(len(gamma)):
                acc_gamma[c] += gamma[c]
    return {"beta": [v / kept for v in acc_beta],
            "gamma": [v / kept for v in acc_gamma[:len(gamma)]],
            "omega_mean": sum(omega) / n, "n_kept": kept,
            "n_categories": C}


def ordinal_latent_predictor(n, X_E=None, X=None, X_EM=None,
                             Z_L=None, L_g=None):
    """The latent-scale predictors of eq. (7.3)-(7.5), pp.219-221:

      (7.3)  L = X_E beta_E + X beta + X_EM beta_EM + eps
             (environment fixed, markers, marker x environment)
      (7.4)  L = Z_L g + eps, g ~ N(0, sigma2_g G)  -- ordinal GBLUP
      (7.5)  L = X_E beta_E + Z_L g + eps           -- environment
             and genetic effects, no interaction

    Blocks are stacked in the order printed in Table 7.6 p.233, where
    the genetic block enters as Z_L L_g with G = L_g L_g'.
    """
    blocks = []
    if X_E is not None:
        blocks.append(("environments", _mat(X_E)))
    if X is not None:
        blocks.append(("markers", _mat(X)))
    if X_EM is not None:
        blocks.append(("env_x_marker", _mat(X_EM)))
    if Z_L is not None:
        Zg = _mm(_mat(Z_L), _mat(L_g)) if L_g is not None \
            else _mat(Z_L)
        blocks.append(("genetic", Zg))
    if not blocks:
        raise ValueError("the predictor needs at least one block")
    design = [sum((blk[i] for _, blk in blocks), [])
              for i in range(n)]
    return {"design": design,
            "widths": {name: len(blk[0]) for name, blk in blocks},
            "n_columns": len(design[0])}


def multinomial_probabilities(X, beta0, beta, baseline_last=True):
    """eq. (7.6) p.225: P(Y_i = c | x_i) = exp(beta_0c + x_i'beta_c)
    / sum_l exp(beta_0l + x_i'beta_l).  The book fixes
    (beta_0C, beta_C) = (0, 0) for identifiability (p.225), which
    ``baseline_last`` applies to the last category."""
    Xm = _mat(X)
    b0 = list(_flat(beta0))
    B = [list(map(float, row)) for row in beta]
    if baseline_last:
        b0 = b0 + [0.0]
        B = B + [[0.0] * len(Xm[0])]
    out = []
    for row in Xm:
        eta = [b0[c] + sum(a * b for a, b in zip(row, B[c]))
               for c in range(len(b0))]
        m = max(eta)
        ex = [math.exp(v - m) for v in eta]
        s = sum(ex)
        out.append([v / s for v in ex])
    return out


def multinomial_loglik(X, y, beta0, beta, baseline_last=True):
    """eq. (7.8) p.226: l(beta; y) = sum_i sum_c 1{y_i = c}
    (beta_0c + x_i'beta_c) - sum_i log sum_l exp(beta_0l
    + x_i'beta_l)."""
    P = multinomial_probabilities(X, beta0, beta, baseline_last)
    ys = [int(v) for v in _flat(y)]
    return sum(math.log(max(P[i][ys[i]], 1e-300))
               for i in range(len(ys)))


def penalized_multinomial_loglik(X, y, beta0, beta, lam,
                                 penalty="ridge",
                                 baseline_last=True):
    """eq. (7.7) p.226 (ridge): l_p = l(beta; y)
    - lambda sum_c beta_c'beta_c, and eq. (7.10) p.227 (lasso):
    l_p = l(beta; y) - lambda sum_c sum_j |beta_cj|.  Only the slopes
    are penalized, never the intercepts (p.226)."""
    ll = multinomial_loglik(X, y, beta0, beta, baseline_last)
    if penalty == "lasso":
        pen = sum(abs(v) for row in beta for v in row)
    else:
        pen = sum(v * v for row in beta for v in row)
    return {"loglik": ll, "penalty": float(lam) * pen,
            "penalized_loglik": ll - float(lam) * pen}


def multinomial_block_update(X, y, beta0, beta, lam, cls,
                             baseline_last=True):
    """eq. (7.9) p.227, the block-coordinate update of class c:
    beta*_c = (X*'W_c X* + lambda D)^-1 X*'W_c y*, where
    X* = [1_n, X], W_c = Diag(w_1c..w_nc) with
    w_ic = p-tilde_c(x_i)(1 - p-tilde_c(x_i)) and the working response
    is y*_ic = beta-tilde_0c + x_i'beta-tilde_c
    + w_ic^-1 (1{y_i = c} - p-tilde_c(x_i)).  D is the identity except
    for a zero in the first entry, so the intercept is unpenalized."""
    Xm = _mat(X)
    n = len(Xm)
    p = len(Xm[0])
    ys = [int(v) for v in _flat(y)]
    P = multinomial_probabilities(Xm, beta0, beta, baseline_last)
    c = int(cls)
    Xs = [[1.0] + row for row in Xm]
    b_cur = [list(_flat(beta0))[c]] + list(map(float, beta[c]))
    w = []
    ystar = []
    for i in range(n):
        pc = min(max(P[i][c], 1e-6), 1.0 - 1e-6)
        wi = pc * (1.0 - pc)
        eta = b_cur[0] + sum(a * b for a, b in zip(Xm[i], b_cur[1:]))
        w.append(wi)
        ystar.append(eta + ((1.0 if ys[i] == c else 0.0) - pc) / wi)
    A = [[sum(w[i] * Xs[i][a] * Xs[i][b] for i in range(n))
          + (float(lam) if (a == b and a > 0) else 0.0)
          for b in range(p + 1)] for a in range(p + 1)]
    rhs = [sum(w[i] * Xs[i][a] * ystar[i] for i in range(n))
           for a in range(p + 1)]
    sol = _solve(A, rhs)
    return {"beta0": sol[0], "beta": sol[1:],
            "weights": w, "working_response": ystar}


def poisson_pmf(y, lam):
    """eq. (7.11) p.232: P(Y = y | x) = lambda^y exp(-lambda)/y! with
    lambda = exp(beta_0 + x'beta)."""
    y = int(y)
    lam = float(lam)
    if lam <= 0:
        raise ValueError("the Poisson mean must be positive")
    return math.exp(y * math.log(lam) - lam - math.lgamma(y + 1.0))


def penalized_poisson_fit(X, y, lam=1.0, penalty="ridge",
                          n_iter=100, tol=1e-10,
                          add_intercept=True):
    """The penalized Poisson log-linear model of sec. 7.5 p.232:
    l_p = sum_i y_i(beta_0 + x_i'beta) - sum_i exp(beta_0 + x_i'beta)
    - sum_i log(y_i!) - (lambda/2) sum_j beta_j^2, fitted by the
    iteratively reweighted least squares the book describes (a
    second-order approximation of the log-likelihood solved as a
    weighted least-squares problem).  The intercept is unpenalized;
    ``penalty='lasso'`` adds a soft-threshold step instead."""
    Xm = _mat(X)
    if add_intercept:
        Xm = [[1.0] + row for row in Xm]
    ys = _flat(y)
    n = len(ys)
    p = len(Xm[0])
    beta = [0.0] * p
    if add_intercept:
        beta[0] = math.log(max(sum(ys) / n, 1e-6))
    for it in range(int(n_iter)):
        eta = _mv(Xm, beta)
        mu = [math.exp(min(v, 700.0)) for v in eta]
        w = [max(m, 1e-9) for m in mu]
        z = [eta[i] + (ys[i] - mu[i]) / w[i] for i in range(n)]
        A = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
              + (float(lam) if (a == b
                                and not (add_intercept and a == 0))
                 else 0.0)
              for b in range(p)] for a in range(p)]
        rhs = [sum(w[i] * Xm[i][a] * z[i] for i in range(n))
               for a in range(p)]
        new = _solve(A, rhs)
        if penalty == "lasso":
            new = [new[0]] + [math.copysign(max(abs(v) - lam, 0.0),
                                            v) for v in new[1:]] \
                if add_intercept else \
                [math.copysign(max(abs(v) - lam, 0.0), v)
                 for v in new]
        gap = max(abs(a - b) for a, b in zip(new, beta))
        beta = new
        if gap < tol:
            break
    eta = _mv(Xm, beta)
    mu = [math.exp(min(v, 700.0)) for v in eta]
    ll = sum(ys[i] * eta[i] - mu[i] - math.lgamma(ys[i] + 1.0)
             for i in range(n))
    pen = float(lam) * sum(
        v * v for j, v in enumerate(beta)
        if not (add_intercept and j == 0)) / 2.0
    return {"beta": beta, "fitted": mu, "loglik": ll,
            "penalized_loglik": ll - pen, "iterations": it + 1}


# ------------------ RKHS regression and kernels (ch. 8 pp.252-255)
def kernel_matrix(X, kernel="linear", gamma=None, degree=2,
                  coef0=1.0, Z=None):
    """The Gram matrix K with K_ij = K(x_i, x_j) (sec. 8.2.2 p.255).

    A kernel is an inner product in an expanded feature space,
    K(x_i, x_j) = phi(x_i)'phi(x_j), so K must be symmetric and
    positive semi-definite; those two properties are what make it
    usable as a covariance in the penalized regression of eq. (8.3).
    ``gamma`` defaults to 1/p for the Gaussian kernel.
    """
    A = _mat(X)
    B = _mat(Z) if Z is not None else A
    p = len(A[0])
    if gamma is None:
        gamma = 1.0 / p
    out = []
    for a in A:
        row = []
        for b in B:
            if kernel == "linear":
                v = sum(u * w for u, w in zip(a, b))
            elif kernel == "gaussian":
                d2 = sum((u - w) ** 2 for u, w in zip(a, b))
                v = math.exp(-gamma * d2)
            elif kernel == "polynomial":
                v = (gamma * sum(u * w for u, w in zip(a, b))
                     + coef0) ** degree
            elif kernel == "exponential":
                d = math.sqrt(sum((u - w) ** 2
                                  for u, w in zip(a, b)))
                v = math.exp(-gamma * d)
            elif kernel == "sigmoid":
                v = math.tanh(gamma * sum(u * w
                                          for u, w in zip(a, b))
                              + coef0)
            else:
                raise ValueError("unknown kernel: %s" % kernel)
            row.append(v)
        out.append(row)
    return out


def is_positive_semidefinite(K, tol=1e-9):
    """Property 2 of p.255: the Gram matrix must be positive
    semi-definite for K to be a valid kernel.  Checked through the
    eigenvalues of the symmetrized matrix."""
    Km = _mat(K)
    n = len(Km)
    S = [[0.5 * (Km[i][j] + Km[j][i]) for j in range(n)]
         for i in range(n)]
    vals, _ = np.linalg.eigh(np.marr(S))
    lam = [float(v) for v in vals._flat()]
    return min(lam) >= -tol, lam


def rkhs_norm(beta, K):
    """The empirical RKHS norm of eq. (8.2) p.254:
    ||f||_H^2 = sum_ij beta_i beta_j K(x_i, x_j) = beta' K beta."""
    b = _flat(beta)
    Km = _mat(K)
    return sum(b[i] * Km[i][j] * b[j]
               for i in range(len(b)) for j in range(len(b)))


def rkhs_predict(K_new, beta, eta0=0.0):
    """The representer-theorem prediction of eq. (8.2) p.254:
    f(x_i) = eta_0 + sum_j beta_j K(x_i, x_j) = eta_0 + k_i'beta."""
    return [float(eta0) + sum(k * b for k, b in zip(row, _flat(beta)))
            for row in _mat(K_new)]


def rkhs_fit_squared_loss(K, y, lam=1.0):
    """eq. (8.3) p.254 with the squared-error loss:
    min over (eta_0, beta) of (1/n) sum (y_i - eta_0 - k_i'beta)^2
    + (lambda/2) beta'K beta.  Setting the gradient to zero gives the
    linear system solved here; with L the squared loss this is the
    kernel ridge / RKHS regression estimator."""
    Km = _mat(K)
    ys = _flat(y)
    n = len(ys)
    A = [[0.0] * (n + 1) for _ in range(n + 1)]
    rhs = [0.0] * (n + 1)
    # d/d eta0: (2/n) sum (eta0 + k_i'beta - y_i) = 0
    A[0][0] = 1.0
    for j in range(n):
        A[0][1 + j] = sum(Km[i][j] for i in range(n)) / n
    rhs[0] = sum(ys) / n
    # d/d beta: (2/n) K'(eta0 1 + K beta - y) + lambda K beta = 0
    KtK = _mm(_t(Km), Km)
    Kty = _mv(_t(Km), ys)
    colsum = [sum(Km[i][j] for i in range(n)) for j in range(n)]
    for a in range(n):
        A[1 + a][0] = colsum[a] * 2.0 / n
        for b in range(n):
            A[1 + a][1 + b] = 2.0 * KtK[a][b] / n \
                + float(lam) * Km[a][b]
        rhs[1 + a] = 2.0 * Kty[a] / n
    sol = _solve(A, rhs)
    eta0, beta = sol[0], sol[1:]
    fitted = rkhs_predict(Km, beta, eta0)
    resid = [a - b for a, b in zip(ys, fitted)]
    return {"eta0": eta0, "beta": beta, "fitted": fitted,
            "residuals": resid,
            "loss": sum(v * v for v in resid) / n,
            "penalty": 0.5 * float(lam) * rkhs_norm(beta, Km),
            "objective": sum(v * v for v in resid) / n
            + 0.5 * float(lam) * rkhs_norm(beta, Km)}


def generalized_kernel_model(K, beta, eta0=0.0, link="identity"):
    """The generalized kernel model of sec. 8.2 p.253:
    y_i ~ p(y_i | mu_i), linear predictor eta_i = f(x_i)
    = eta_0 + k_i'beta, and link eta_i = g(mu_i).  Applies the inverse
    link h = g^-1 to return the mean.  One framework covers continuous
    (identity), binary (logit/probit), count (log) and categorical
    responses."""
    eta = rkhs_predict(K, beta, eta0)
    if link == "identity":
        mu = eta
    elif link == "logit":
        mu = [1.0 / (1.0 + math.exp(-v)) for v in eta]
    elif link == "probit":
        mu = [_norm_cdf(v) for v in eta]
    elif link == "log":
        mu = [math.exp(min(v, 700.0)) for v in eta]
    else:
        raise ValueError("unknown link: %s" % link)
    return {"eta": eta, "mu": mu, "link": link}


def arccos_kernel(X, Z=None, depth=1, normalize_median=False):
    """The arc-cosine kernel of eq. (8.4) p.265 and its deep
    recursion (8.5) p.266 (Cho and Saul 2009).

    With theta_ij = arccos(x_i'x_j / (||x_i|| ||x_j||)) and
    J(theta) = sin(theta) + (pi - theta) cos(theta),

        AK^1(x_i, x_j) = (1/pi) ||x_i|| ||x_j|| J(theta_ij)

    which is positive semi-definite and corresponds to a
    single-hidden-layer network with a ramp activation.  Repeating the
    interior product emulates additional hidden layers:

        AK^(l+1)(x_i, x_j) = (1/pi)
            [AK^l(x_i,x_i) AK^l(x_j,x_j)]^(1/2) J(theta^l_ij),
        theta^l_ij = arccos{AK^l(x_i,x_j)
                            [AK^l(x_i,x_i) AK^l(x_j,x_j)]^(-1/2)}

    so no bandwidth parameter is needed -- only the number of layers.
    ``normalize_median`` divides by the median entry, as the book's R
    code does.
    """
    A = _mat(X)
    B = _mat(Z) if Z is not None else A
    same = Z is None

    def _norms(M):
        return [math.sqrt(sum(v * v for v in row)) for row in M]

    na, nb = _norms(A), _norms(B)

    def J(th):
        return math.sin(th) + (math.pi - th) * math.cos(th)

    def ak1(u, v, nu, nv):
        if nu <= 0 or nv <= 0:
            return 0.0
        c = sum(a * b for a, b in zip(u, v)) / (nu * nv)
        c = min(max(c, -1.0), 1.0)
        return nu * nv * J(math.acos(c)) / math.pi

    K = [[ak1(a, b, na[i], nb[j]) for j, b in enumerate(B)]
         for i, a in enumerate(A)]
    if same:
        dA = [K[i][i] for i in range(len(A))]
        dB = dA
    else:
        dA = [ak1(a, a, na[i], na[i]) for i, a in enumerate(A)]
        dB = [ak1(b, b, nb[j], nb[j]) for j, b in enumerate(B)]
    for _ in range(int(depth) - 1):
        newK = []
        for i in range(len(A)):
            row = []
            for j in range(len(B)):
                den = math.sqrt(max(dA[i] * dB[j], 0.0))
                if den <= 0:
                    row.append(0.0)
                    continue
                c = min(max(K[i][j] / den, -1.0), 1.0)
                row.append(den * J(math.acos(c)) / math.pi)
            newK.append(row)
        dA = [dA[i] * J(0.0) / math.pi for i in range(len(dA))]
        dB = [dB[j] * J(0.0) / math.pi for j in range(len(dB))]
        K = newK
    if normalize_median:
        flat = sorted(v for row in K for v in row)
        n = len(flat)
        med = flat[n // 2] if n % 2 else 0.5 * (flat[n // 2 - 1]
                                                + flat[n // 2])
        if med:
            K = [[v / med for v in row] for row in K]
    return K


def hadamard(A, B):
    """Element-wise (Hadamard) product, written "o" on p.285."""
    Am, Bm = _mat(A), _mat(B)
    return [[Am[i][j] * Bm[i][j] for j in range(len(Am[0]))]
            for i in range(len(Am))]


def bayesian_kernel_blup(y, K, sigma2_u=1.0, sigma2_e=1.0,
                         X=None, n_iter=1200, burn_in=300,
                         nu=5.0, nu_u=5.0, R2=0.5, seed=42,
                         gibbs=True):
    """The Bayesian kernel BLUP of eq. (8.8) p.281:
    y = 1 mu + u + e with u ~ N(0, sigma2_u K) and
    e ~ N(0, sigma2_e I), i.e. kernel ridge regression under a
    Bayesian framework with lambda = sigma2_e/sigma2_u.

    Full conditionals (p.282):
      u       | . ~ N(u-tilde, K-tilde),
                    K-tilde = (sigma_u^-2 K^-1 + sigma_e^-2 I)^-1,
                    u-tilde = sigma_e^-2 K-tilde (y - 1 mu)
      mu      | . ~ N((1/n) 1'(y - u), sigma2_e/n)
      sigma2_e| . ~ chi^-2(nu + n, S + ||y - 1 mu - u||^2)
      sigma2_u| . ~ chi^-2(nu_u + n, S_u + u' K^-1 u)

    The mean of u is the BLUP from Henderson's mixed model equation,
    which is why the book notes that with K the genomic relationship
    matrix this model IS GBLUP (p.282).  ``gibbs=False`` returns the
    single conditional-mode solution at the supplied variances.
    """
    ys = _flat(y)
    n = len(ys)
    Km = _mat(K)
    Kinv = _inv(Km)
    rng = np.random.default_rng(seed)
    hp = brr_hyperparameters(ys, R2=R2, nu=nu, nu_beta=nu_u)

    def posterior_u(mu, s2u, s2e):
        A = [[Kinv[i][j] / s2u + ((1.0 / s2e) if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        Kt = _inv(A)
        ut = [v / s2e for v in _mv(Kt, [a - mu for a in ys])]
        return ut, Kt

    if not gibbs:
        mu = sum(ys) / n
        ut, Kt = posterior_u(mu, sigma2_u, sigma2_e)
        return {"mu": mu, "u": ut, "K_tilde": Kt,
                "sigma2_u": sigma2_u, "sigma2_e": sigma2_e}

    mu = sum(ys) / n
    s2u, s2e = sigma2_u, sigma2_e
    u = [0.0] * n
    acc_mu = acc_u = acc_s2u = acc_s2e = None
    acc_u = [0.0] * n
    acc_mu = acc_s2u = acc_s2e = 0.0
    kept = 0
    for it in range(int(n_iter)):
        ut, Kt = posterior_u(mu, s2u, s2e)
        L = _chol(Kt)
        z = [float(rng.normal(0, 1)) for _ in range(n)]
        u = [ut[i] + sum(L[i][k] * z[k] for k in range(n))
             for i in range(n)]
        resid = [a - b for a, b in zip(ys, u)]
        mu = sum(resid) / n + math.sqrt(s2e / n) \
            * float(rng.normal(0, 1))
        e = [a - mu for a in resid]
        s2e = scaled_inv_chisq(rng, nu + n,
                               hp["S"] + sum(v * v for v in e))
        quad = sum(u[i] * sum(Kinv[i][j] * u[j] for j in range(n))
                   for i in range(n))
        s2u = scaled_inv_chisq(rng, nu_u + n,
                               hp["S_beta"] + quad)
        if it >= burn_in:
            kept += 1
            acc_mu += mu
            acc_s2u += s2u
            acc_s2e += s2e
            for i in range(n):
                acc_u[i] += u[i]
    return {"mu": acc_mu / kept, "u": [v / kept for v in acc_u],
            "sigma2_u": acc_s2u / kept, "sigma2_e": acc_s2e / kept,
            "n_kept": kept}


def kernel_blup_replicated(Z, K, sigma2_u=1.0):
    """eq. (8.9) p.282: Y = 1 mu + Z u + e when individuals are
    replicated.  BGLR cannot take that predictor directly, so the
    covariance of the predictor is precomputed,
    K_* = Var(Z u) = Z K Z', and used as the kernel."""
    Zm = _mat(Z)
    Ks = _mm(_mm(Zm, _mat(K)), _t(Zm))
    return [[sigma2_u * v for v in row] for row in Ks]


def kernel_blup_gxe(Z_u1, K, Z_E, sigma2_u1=1.0, sigma2_u2=1.0):
    """eq. (8.10) p.283-285, the extended predictor:
    y = mu 1 + Z_E beta_E + u_1 + u_2 + eps with
    u_1 ~ N(0, sigma2_u1 K_1), K_1 = Z_u1 K Z_u1', and
    u_2 ~ N(0, sigma2_u2 K_2),
    K_2 = (Z_u1 K Z_u1') o (Z_E Z_E'), "o" the Hadamard product.
    K_1 carries the genomic main effects and K_2 the genotype x
    environment interaction."""
    Zu = _mat(Z_u1)
    ZE = _mat(Z_E)
    K1 = _mm(_mm(Zu, _mat(K)), _t(Zu))
    KE = _mm(ZE, _t(ZE))
    K2 = hadamard(K1, KE)
    return {"K1": [[sigma2_u1 * v for v in row] for row in K1],
            "K2": [[sigma2_u2 * v for v in row] for row in K2],
            "K_env": KE}


def kernel_eigen_design(K, tol=1e-10):
    """eq. (8.11) p.289: with the eigendecomposition
    K = U S^(1/2) S^(1/2) U', model (8.8) is reparameterized as
    y = mu 1_n + P f + eps with f ~ N(0, sigma2_f I_r) and
    P = U S^(1/2), r = rank(K).  Because P P' = K the two models are
    equivalent, but (8.11) is an ordinary ridge regression on r
    columns, and r is usually far below min(n, p).
    """
    Km = _mat(K)
    n = len(Km)
    S = [[0.5 * (Km[i][j] + Km[j][i]) for j in range(n)]
         for i in range(n)]
    vals, vecs = np.linalg.eigh(np.marr(S))
    lam = [float(v) for v in vals._flat()]
    V = [[float(v) for v in row] for row in vecs._tolist()] \
        if hasattr(vecs, "_tolist") else _mat(vecs)
    order = sorted(range(n), key=lambda i: -lam[i])
    keep = [i for i in order if lam[i] > tol]
    P = [[V[r][i] * math.sqrt(lam[i]) for i in keep]
         for r in range(n)]
    return {"P": P, "rank": len(keep),
            "eigenvalues": [lam[i] for i in keep]}


def nystrom_kernel(X, m_index, kernel="linear", gamma=None):
    """The Nystrom approximation of p.290 (Williams and Seeger 2001)
    as used by Cuevas et al. (2020):

        K ~= Q = K_{n,m} K_{m,m}^-1 K_{n,m}'

    built from m of the n lines, so only K_{m,m} and K_{n,m} are ever
    formed.  For the linear kernel K_{m,m} = X_m X_m'/p and
    K_{n,m} = X_n X_m'/p.  Q has rank m and equals K exactly when the
    m rows span the same space, in particular when m = n.
    """
    A = _mat(X)
    idx = [int(i) for i in m_index]
    Xm = [A[i] for i in idx]
    p = len(A[0])
    if kernel == "linear":
        Kmm = [[sum(a * b for a, b in zip(u, v)) / p for v in Xm]
               for u in Xm]
        Knm = [[sum(a * b for a, b in zip(u, v)) / p for v in Xm]
               for u in A]
    else:
        Kmm = kernel_matrix(Xm, kernel=kernel, gamma=gamma)
        Knm = kernel_matrix(A, kernel=kernel, gamma=gamma, Z=Xm)
    Kmm_inv = _inv(Kmm)
    Q = _mm(_mm(Knm, Kmm_inv), _t(Knm))
    return {"Q": Q, "K_mm": Kmm, "K_nm": Knm, "rank": len(idx)}


def sparse_kernel_design(X, m_index, kernel="linear", gamma=None,
                         tol=1e-10):
    """eq. (8.12) p.291, steps 1-5: build K_{m,m} from m training
    lines, K_{n,m} against all n, take the eigendecomposition
    K_{m,m}^-1 = U S^(-1/2) S^(-1/2) U', and form the design
    P = K_{n,m} U S^(-1/2).  Then y = mu 1_n + P f + eps with
    f ~ N(0, sigma2_f I_m) is an ordinary ridge regression, and
    P P' reproduces the Nystrom approximation Q.
    """
    ny = nystrom_kernel(X, m_index, kernel=kernel, gamma=gamma)
    Kmm = ny["K_mm"]
    m = len(Kmm)
    S = [[0.5 * (Kmm[i][j] + Kmm[j][i]) for j in range(m)]
         for i in range(m)]
    vals, vecs = np.linalg.eigh(np.marr(S))
    lam = [float(v) for v in vals._flat()]
    V = [[float(v) for v in row] for row in vecs._tolist()] \
        if hasattr(vecs, "_tolist") else _mat(vecs)
    order = sorted(range(m), key=lambda i: -lam[i])
    keep = [i for i in order if lam[i] > tol]
    # U S^(-1/2) using the eigenvalues of K_mm
    US = [[V[r][i] / math.sqrt(lam[i]) for i in keep]
          for r in range(m)]
    P = _mm(ny["K_nm"], US)
    return {"P": P, "Q": ny["Q"], "rank": len(keep),
            "K_mm": Kmm, "K_nm": ny["K_nm"]}


def rkhs_mixed_equations(C, K, y, lam=1.0, sigma2_e=1.0,
                         form="direct"):
    """The RKHS regression estimating equations of sec. 8.6 p.276.

    Minimizing J[theta, beta | lambda] = (1/(2 sigma2_e))
    (y - C theta - K beta)'(y - C theta - K beta)
    + (lambda/2) beta' K beta and setting both gradients to zero gives

      (8.6)  [C'C   C'K          ] [theta]   [C'y]
             [K'C   K'K + lambda K sigma2_e] [beta ] = [K'y]

    Because K is symmetric, K'K = K^2; multiplying the second block by
    K^-1 gives the equivalent but cheaper system

      (8.7)  [C'C   C'K              ] [theta]   [C'y]
             [I'C   K + lambda I sigma2_e] [beta ] = [ y ]

    which avoids inverting K and forming K'K.  ``form='direct'`` uses
    (8.6), ``form='reduced'`` uses (8.7); the book states the two give
    the same solution.  sigma2_beta = 1/lambda is read as the variation
    due to marked additive genomic variation.
    """
    Cm = _mat(C)
    Km = _mat(K)
    ys = _flat(y)
    n = len(ys)
    q = len(Cm[0])
    Ct = _t(Cm)
    CtC = _mm(Ct, Cm)
    CtK = _mm(Ct, Km)
    if form == "direct":
        KtC = _mm(_t(Km), Cm)
        KtK = _mm(_t(Km), Km)
        A22 = [[KtK[i][j] + lam * Km[i][j] * sigma2_e
                for j in range(n)] for i in range(n)]
        rhs = _mv(Ct, ys) + _mv(_t(Km), ys)
        top = [CtC[i] + CtK[i] for i in range(q)]
        bot = [KtC[i] + A22[i] for i in range(n)]
    elif form == "reduced":
        A22 = [[Km[i][j] + (lam * sigma2_e if i == j else 0.0)
                for j in range(n)] for i in range(n)]
        rhs = _mv(Ct, ys) + ys
        top = [CtC[i] + CtK[i] for i in range(q)]
        bot = [list(Cm[i]) + A22[i] for i in range(n)]
    else:
        raise ValueError("form must be 'direct' or 'reduced'")
    sol = _solve(top + bot, rhs)
    theta, beta = sol[:q], sol[q:]
    u = _mv(Km, beta)                 # reparameterization II: u = K beta
    fitted = [a + b for a, b in zip(_mv(Cm, theta), u)]
    return {"theta": theta, "beta": beta, "u": u, "fitted": fitted,
            "sigma2_beta": 1.0 / lam if lam else float("inf")}


def rkhs_predict_new(K_star, beta):
    """p.276: breeding values for t new genotyped individuals without
    phenotype follow from a matrix-vector product, u_new = K_s beta,
    where K_s is the t x n matrix of genomic similarities between the
    new individuals and the training set."""
    return _mv(_mat(K_star), _flat(beta))


# ---------------- support vector machines (ch. 9 pp.339-350)
def hyperplane_side(X, beta0, beta):
    """eq. (9.4) p.340: a point lies on one side of the hyperplane
    when beta_0 + beta_1 x_1 + ... + beta_p x_p > 0 and on the other
    when it is < 0, so the sign of the left-hand side says which."""
    B = _flat(beta)
    return [1 if (float(beta0) + sum(a * b for a, b in zip(row, B)))
            > 0 else -1 for row in _mat(X)]


def svm_decision_values(X, beta0, beta):
    """eq. (9.5) p.341: f(x_i) = beta_0 + x_i' beta, the fitting
    function whose sign classifies a new observation and whose
    magnitude expresses confidence."""
    B = _flat(beta)
    return [float(beta0) + sum(a * b for a, b in zip(row, B))
            for row in _mat(X)]


def svm_label_matrix(X, y):
    """The matrix Q of Example 9.1 p.350, whose rows are y_i x_i.
    The dual objective depends on the data only through Q Q', i.e.
    through inner products of the labelled vectors."""
    Xm = _mat(X)
    ys = _flat(y)
    return [[ys[i] * v for v in Xm[i]] for i in range(len(Xm))]


def svm_dual_objective(alpha, X, y, K=None):
    """eq. (9.32) p.349: L(alpha) = sum_i alpha_i
    - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j).
    The dual depends on the data only through inner products, which
    is exactly what lets a kernel replace them."""
    a = _flat(alpha)
    ys = _flat(y)
    n = len(a)
    G = _mat(K) if K is not None else \
        _mm(_mat(X), _t(_mat(X)))
    quad = sum(a[i] * a[j] * ys[i] * ys[j] * G[i][j]
               for i in range(n) for j in range(n))
    return sum(a) - 0.5 * quad


def svm_dual_constraints_ok(alpha, y, C=None, tol=1e-8):
    """eq. (9.33) p.349: alpha_i >= 0 and sum_i alpha_i y_i = 0 (with
    an upper bound C in the soft-margin case)."""
    a = _flat(alpha)
    ys = _flat(y)
    nonneg = all(v >= -tol for v in a)
    bounded = True if C is None else all(v <= C + tol for v in a)
    balanced = abs(sum(ai * yi for ai, yi in zip(a, ys))) < 1e-6
    return {"nonnegative": nonneg, "bounded": bounded,
            "balanced": balanced,
            "feasible": nonneg and bounded and balanced}


def svm_beta_from_alpha(alpha, X, y):
    """eq. (9.28) p.348: setting dL/dbeta = 0 gives
    beta = sum_i alpha_i y_i x_i, so the weights are a linear
    combination of the training vectors; only those with alpha_i != 0
    (the support vectors) contribute."""
    a = _flat(alpha)
    ys = _flat(y)
    Xm = _mat(X)
    p = len(Xm[0])
    return [sum(a[i] * ys[i] * Xm[i][j] for i in range(len(a)))
            for j in range(p)]


def svm_intercept(alpha, X, y, K=None, tol=1e-8):
    """p.350: averaging y_i(beta_0 + x_i'beta) = 1 over the support
    vectors gives the numerically stable intercept
    beta_0 = (1/N_S) sum_{i in S} (y_i
             - sum_{j in S} alpha_j y_j (x_i . x_j))."""
    a = _flat(alpha)
    ys = _flat(y)
    n = len(a)
    G = _mat(K) if K is not None else _mm(_mat(X), _t(_mat(X)))
    S = [i for i in range(n) if a[i] > tol]
    if not S:
        return 0.0
    return sum(ys[i] - sum(a[j] * ys[j] * G[i][j] for j in S)
               for i in S) / len(S)


def svm_fit_dual(X, y, C=None, n_iter=4000, tol=1e-9, K=None,
                 lr=None):
    """Maximize the dual (9.32) subject to (9.33) by projected
    gradient ascent, keeping sum_i alpha_i y_i = 0 on every step.

    The gradient is dL/dalpha_i = 1 - y_i sum_j alpha_j y_j G_ij; the
    balance constraint of (9.33) is maintained by removing the
    component of the gradient along y before stepping, and alpha is
    clipped to [0, C].  Returns alpha, the weights (9.28), the
    intercept of p.350, and the support-vector indices.
    """
    Xm = _mat(X)
    ys = _flat(y)
    n = len(ys)
    G = _mat(K) if K is not None else _mm(Xm, _t(Xm))
    H = [[ys[i] * ys[j] * G[i][j] for j in range(n)]
         for i in range(n)]
    scale = max(abs(H[i][i]) for i in range(n)) or 1.0
    step = (lr if lr is not None else 1.0 / (n * scale))
    a = [0.0] * n
    yy = sum(v * v for v in ys)
    for _ in range(int(n_iter)):
        grad = [1.0 - sum(H[i][j] * a[j] for j in range(n))
                for i in range(n)]
        # project the gradient onto {g : g . y = 0} (eq. 9.33)
        gy = sum(g * v for g, v in zip(grad, ys)) / yy
        grad = [g - gy * v for g, v in zip(grad, ys)]
        new = [a[i] + step * grad[i] for i in range(n)]
        new = [max(0.0, v if C is None else min(v, C)) for v in new]
        if max(abs(new[i] - a[i]) for i in range(n)) < tol:
            a = new
            break
        a = new
    beta = svm_beta_from_alpha(a, Xm, ys)
    b0 = svm_intercept(a, Xm, ys, K=K)
    sv = [i for i in range(n) if a[i] > 1e-6]
    return {"alpha": a, "beta": beta, "beta0": b0,
            "support_vectors": sv,
            "objective": svm_dual_objective(a, Xm, ys, K=K)}


def svm_predict(X_new, beta0, beta):
    """p.350: y-hat = sign[f-hat(x)] with f-hat from eq. (9.5)."""
    return [1 if v > 0 else -1
            for v in svm_decision_values(X_new, beta0, beta)]


# ------------- artificial neural networks (ch. 10 pp.385, 409-412)
def _act(name, z, deriv=False):
    """Activation functions and their derivatives, used as g^(h) and
    g^(l) in eq. (10.7) and (10.9)."""
    if name == "identity":
        return 1.0 if deriv else z
    if name == "logistic":
        s = 1.0 / (1.0 + math.exp(-max(min(z, 700.0), -700.0)))
        return s * (1.0 - s) if deriv else s
    if name == "tanh":
        t = math.tanh(z)
        return 1.0 - t * t if deriv else t
    if name == "relu":
        return (1.0 if z > 0 else 0.0) if deriv else max(0.0, z)
    raise ValueError("unknown activation: %s" % name)


def ann_forward(X, W, activations=None):
    """eq. (10.1)-(10.3) p.385: a feedforward pass through a network
    with d inputs, M_1 units in hidden layer 1, M_2 in hidden layer 2
    and O outputs,

        V_1j = g_1(sum_i w_ji^(1) x_i)
        V_2k = g_2(sum_j w_kj^(2) V_1j)
        y_l  = g_3(sum_k w_lk^(3) V_2k)

    generalized to any number of layers.  ``W`` is a list of weight
    matrices, one per layer, each shaped (units_out x units_in); the
    bias is handled by adding an input fixed at 1 (p.409).
    """
    A = _mat(X)
    acts = activations or ["logistic"] * (len(W) - 1) + ["identity"]
    layers = [A]
    nets = []
    for li, Wl in enumerate(W):
        Wm = _mat(Wl)
        z = [[sum(Wm[u][v] * row[v] for v in range(len(Wm[0])))
              for u in range(len(Wm))] for row in layers[-1]]
        nets.append(z)
        layers.append([[_act(acts[li], v) for v in row]
                       for row in z])
    return {"output": layers[-1], "layers": layers, "nets": nets,
            "activations": acts}


def ann_sse(y_hat, y):
    """eq. (10.5) p.409: E = (1/2) sum_i sum_j (y-hat_ij - y_ij)^2,
    the sum-of-squared-errors loss the backpropagation derivation
    minimizes."""
    P = _mat(y_hat)
    Y = _mat(y)
    return 0.5 * sum((P[i][j] - Y[i][j]) ** 2
                     for i in range(len(P)) for j in range(len(P[0])))


def ann_backprop_gradients(X, y, W, activations=None):
    """The backpropagation gradients of eq. (10.10)-(10.17) p.410-412.

    Output layer, eq. (10.12): delta_ij = (y_ij - y-hat_ij)
    g^(l)'(z_ij^(l)) and Delta w_jk^(l) = eta delta_ij V_ik^(h).
    Hidden layer, eq. (10.16): psi_ik = sum_j delta_ij w_jk^(l)
    g^(h)'(z_ik^(h)) and Delta w_kp^(h) = eta psi_ik x_ip -- the sum
    over outputs is there because every hidden neuron feeds all of
    them.

    Returns dE/dW per layer (note eq. 10.10 defines the UPDATE as
    -eta times this, so the deltas above carry the opposite sign).
    """
    fwd = ann_forward(X, W, activations)
    layers, nets, acts = fwd["layers"], fwd["nets"], fwd["activations"]
    Y = _mat(y)
    n = len(Y)
    L = len(W)
    # delta at the output: dE/dz = (y-hat - y) g'(z)
    d = [[(layers[-1][i][j] - Y[i][j])
          * _act(acts[-1], nets[-1][i][j], deriv=True)
          for j in range(len(Y[0]))] for i in range(n)]
    grads = [None] * L
    for li in range(L - 1, -1, -1):
        Wm = _mat(W[li])
        prev = layers[li]
        grads[li] = [[sum(d[i][u] * prev[i][v] for i in range(n))
                      for v in range(len(Wm[0]))]
                     for u in range(len(Wm))]
        if li > 0:
            d = [[sum(d[i][u] * Wm[u][v] for u in range(len(Wm)))
                  * _act(acts[li - 1], nets[li - 1][i][v],
                         deriv=True)
                  for v in range(len(Wm[0]))] for i in range(n)]
    return {"gradients": grads, "loss": ann_sse(layers[-1], Y),
            "output": layers[-1]}


def ann_train(X, y, W, eta=0.1, n_iter=500, activations=None,
              tol=1e-12):
    """The weight updates of eq. (10.13) and (10.17):
    w^(t+1) = w^(t) + eta delta V and w^(t+1) = w^(t) + eta psi x,
    iterated until the loss stops decreasing."""
    Wc = [[list(map(float, row)) for row in _mat(Wl)] for Wl in W]
    hist = []
    for it in range(int(n_iter)):
        g = ann_backprop_gradients(X, y, Wc, activations)
        hist.append(g["loss"])
        for li in range(len(Wc)):
            for u in range(len(Wc[li])):
                for v in range(len(Wc[li][u])):
                    Wc[li][u][v] -= eta * g["gradients"][li][u][v]
        if len(hist) > 1 and abs(hist[-2] - hist[-1]) < tol:
            break
    final = ann_backprop_gradients(X, y, Wc, activations)
    return {"W": Wc, "loss": final["loss"], "history": hist,
            "output": final["output"], "iterations": len(hist)}


def ann_numeric_gradient(X, y, W, activations=None, eps=1e-6):
    """Central-difference gradient of eq. (10.5), used to check the
    analytic backpropagation gradients."""
    out = []
    for li in range(len(W)):
        Wm = _mat(W[li])
        G = [[0.0] * len(Wm[0]) for _ in range(len(Wm))]
        for u in range(len(Wm)):
            for v in range(len(Wm[0])):
                for sign in (1, -1):
                    Wp = [[list(map(float, r)) for r in _mat(A)]
                          for A in W]
                    Wp[li][u][v] += sign * eps
                    e = ann_sse(ann_forward(X, Wp,
                                            activations)["output"],
                                y)
                    G[u][v] += sign * e
                G[u][v] /= (2.0 * eps)
        out.append(G)
    return out


# ------------------- convolutional networks (ch. 13 p.551)
def conv2d(image, kernel, bias=0.0, stride=1, activation=None):
    """eq. (13.1)-(13.2) p.551: the filter slides over the image and
    at each position computes the dot product of the local receptive
    field with the filter, z = w'x + b, then (13.2) applies the
    activation to give the feature (activation) map.

    A 7 x 7 x 3 filter on a 256 x 256 x 3 image needs only
    7*7*3 + 1 = 148 parameters instead of the 256*256*3 + 1 = 196,609
    a fully connected layer would need, because the weights are shared
    across positions.  That sharing is also what gives CNNs
    translational invariance: the same filter detects the feature
    wherever it appears.  Output size is (H - F)/stride + 1.
    """
    I = [[list(map(float, ch)) for ch in row] for row in image] \
        if isinstance(image[0][0], (list, tuple)) else \
        [[[float(v)] for v in row] for row in image]
    K = [[list(map(float, ch)) for ch in row] for row in kernel] \
        if isinstance(kernel[0][0], (list, tuple)) else \
        [[[float(v)] for v in row] for row in kernel]
    H, W, C = len(I), len(I[0]), len(I[0][0])
    F, Fw = len(K), len(K[0])
    out_h = (H - F) // stride + 1
    out_w = (W - Fw) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("filter is larger than the image")
    out = []
    for r in range(out_h):
        row = []
        for c in range(out_w):
            z = float(bias)
            for a in range(F):
                for b in range(Fw):
                    for ch in range(C):
                        z += K[a][b][ch] * \
                            I[r * stride + a][c * stride + b][ch]
            row.append(_act(activation, z) if activation else z)
        out.append(row)
    return out


def conv_output_size(input_size, filter_size, stride=1, padding=0):
    """p.551: a 256 x 256 image with a 7 x 7 filter gives a 250 x 250
    feature map, i.e. (H - F + 2P)/S + 1."""
    return (int(input_size) - int(filter_size)
            + 2 * int(padding)) // int(stride) + 1


def conv_parameter_count(filter_size, channels, n_filters=1):
    """p.551: F*F*C + 1 parameters per filter, against
    H*W*C + 1 for a fully connected layer -- 148 versus 196,609 in the
    book's example."""
    return n_filters * (int(filter_size) ** 2 * int(channels) + 1)


# --------------- functional regression (ch. 14 pp.580-583)
def fda_basis_matrix(t, n_basis, kind="fourier", period=None):
    """The basis matrix Psi of eq. (14.8) p.581, with rows indexed by
    the observation times t_1 < ... < t_m and columns by the basis
    functions psi_1..psi_L2."""
    ts = _flat(t)
    m = len(ts)
    if m == 0:
        raise ValueError("t is empty.")
    L = int(n_basis)
    if L < 1:
        raise ValueError("n_basis must be a positive integer; got %r"
                         % (n_basis,))
    lo, hi = min(ts), max(ts)
    span = (hi - lo) or 1.0
    if period is None:
        P = span
    else:
        # `float(period) if period else span` treated an explicit 0 as
        # "not supplied" and passed a negative period straight into the
        # sine argument. R clamped non-positive to 1. Both now refuse it.
        P = float(period)
        if not (P > 0.0) or P != P or P in (float("inf"), float("-inf")):
            raise ValueError("period must be a finite positive number; "
                             "got %r" % (period,))
    out = []
    for tv in ts:
        row = []
        for l in range(L):
            if kind == "fourier":
                if l == 0:
                    row.append(1.0)
                elif l % 2 == 1:
                    row.append(math.sin(2.0 * math.pi
                                        * ((l + 1) // 2) * tv / P))
                else:
                    row.append(math.cos(2.0 * math.pi
                                        * (l // 2) * tv / P))
            elif kind in ("poly", "polynomial"):
                # "poly" is the spelling R's own @param documents; Python
                # accepted only "polynomial" and raised on the other.
                row.append(((tv - lo) / span) ** l)
            else:
                raise ValueError("unknown basis: %s" % kind)
        out.append(row)
    return out


def fda_basis_coefficients(Psi, x_t):
    """eq. (14.7) p.581: c-hat_i = (Psi'Psi)^-1 Psi' x_i(t), the least
    squares (equivalently maximum likelihood) coefficients of the
    expansion x_i(t) = sum_o c_io psi_o(t) of eq. (14.6)."""
    P = _mat(Psi)
    Pt = _t(P)
    return _solve(_mm(Pt, P), _mv(Pt, _flat(x_t)))


def fda_inner_product_matrix(t, L1, L2, kind="fourier"):
    """The matrix Q of p.581 whose (l, o) entry is
    int_0^T phi_l(t) psi_o(t) dt, computed by the trapezoid rule on
    the observation grid."""
    ts = _flat(t)
    Phi = fda_basis_matrix(ts, L1, kind=kind)
    Psi = fda_basis_matrix(ts, L2, kind=kind)
    m = len(ts)
    Q = [[0.0] * L2 for _ in range(L1)]
    for l in range(L1):
        for o in range(L2):
            s = 0.0
            for j in range(m - 1):
                dt = ts[j + 1] - ts[j]
                s += 0.5 * dt * (Phi[j][l] * Psi[j][o]
                                 + Phi[j + 1][l] * Psi[j + 1][o])
            Q[l][o] = s
    return Q


def fda_design_matrix(t, X_curves, L1, L2, kind="fourier"):
    """eq. (14.9) p.581-582: X* = [1_n  X] with
    X = X** Psi (Psi'Psi)^-1 Q', so each row is x_i = Q c-hat_i, the
    functional covariate reduced to L1 scalar scores."""
    ts = _flat(t)
    Psi = fda_basis_matrix(ts, L2, kind=kind)
    Q = fda_inner_product_matrix(ts, L1, L2, kind=kind)
    rows = []
    for curve in _mat(X_curves):
        c = fda_basis_coefficients(Psi, curve)
        rows.append(_mv(Q, c))
    return {"X": rows,
            "X_star": [[1.0] + r for r in rows],
            "Q": Q, "Psi": Psi}


def fda_fit(t, X_curves, y, L1=3, L2=5, kind="fourier"):
    """eq. (14.3)-(14.5) pp.580: with beta(t) expanded on L1 bases the
    functional model becomes Y = x*'beta + eps, so
    beta-hat = (X*'X*)^-1 X*'y (14.4) and
    sigma2-hat = (1/n)(y - X*beta-hat)'(y - X*beta-hat) (14.5).
    beta-hat(t) = sum_l beta-hat_l phi_l(t) recovers the coefficient
    function (14.2)."""
    d = fda_design_matrix(t, X_curves, L1, L2, kind=kind)
    Xs = d["X_star"]
    ys = _flat(y)
    n = len(ys)
    beta = _solve(_mm(_t(Xs), Xs), _mv(_t(Xs), ys))
    fitted = _mv(Xs, beta)
    resid = [a - b for a, b in zip(ys, fitted)]
    s2 = sum(v * v for v in resid) / n
    return {"beta": beta, "fitted": fitted, "residuals": resid,
            "sigma2": s2, "X_star": Xs, "Q": d["Q"]}


def fda_beta_function(t, beta_coefs, L1, kind="fourier"):
    """eq. (14.2): beta-hat(t) = sum_l beta-hat_l phi_l(t), the
    basis-based estimate of the coefficient function."""
    Phi = fda_basis_matrix(t, L1, kind=kind)
    b = _flat(beta_coefs)
    return [sum(Phi[j][l] * b[l] for l in range(L1))
            for j in range(len(Phi))]


def fda_bic(loglik, n_params, n_obs):
    """p.582: BIC = -2 l(beta-hat, sigma2-hat; y) + (L + 1) log(n);
    the basis size with the lowest BIC is preferred."""
    return -2.0 * float(loglik) + (int(n_params) + 1) \
        * math.log(int(n_obs))


def fda_loocv(t, x_t, L2, kind="fourier"):
    """p.583: CV_1(L2) = sum_j (x(t_j) - x-hat_-j(t_j))^2, the
    leave-one-out criterion for choosing the number of basis
    functions representing the covariate curve."""
    ts = _flat(t)
    xs = _flat(x_t)
    m = len(ts)
    # The basis functions are fixed on the domain of the curve; only
    # the coefficients are re-estimated when a point is held out.  The
    # period must therefore come from the full grid -- deriving it from
    # the reduced grid would change phi_l itself whenever the dropped
    # point is an endpoint, and the held-out value would then be
    # predicted from a different basis than the one it was fitted on.
    period = max(ts) - min(ts)
    if period <= 0:
        period = 1.0
    tot = 0.0
    for j in range(m):
        t_j = ts[:j] + ts[j + 1:]
        x_j = xs[:j] + xs[j + 1:]
        Psi_j = fda_basis_matrix(t_j, L2, kind=kind, period=period)
        c = fda_basis_coefficients(Psi_j, x_j)
        psi_at_j = fda_basis_matrix([ts[j]], L2, kind=kind,
                                    period=period)[0]
        pred = sum(psi_at_j[o] * c[o] for o in range(L2))
        tot += (xs[j] - pred) ** 2
    return tot


# ------- random forest for count responses (ch. 15 pp.651-653)
def zap_link(mu_pred, theta_pred):
    """eq. (15.1) p.651: log(mu) = f_mu(x) and
    log(theta/(1-theta)) = f_theta(x), the nonparametric link
    functions the ZAP random forest estimates in two steps."""
    mu = math.exp(min(float(mu_pred), 700.0))
    theta = 1.0 / (1.0 + math.exp(-float(theta_pred)))
    return {"mu": mu, "theta": theta}


def zero_truncated_poisson_loglik(y_positive, mu):
    """eq. (15.2) p.651: LL+ = -N+ log(1 - exp(-mu))
    + log(mu) sum_i Y_i+ - N+ mu - sum_i log(Y_i+!), the
    zero-truncated Poisson log-likelihood used as the splitting
    criterion in the truncated part of the forest."""
    ys = _flat(y_positive)
    n = len(ys)
    mu = float(mu)
    if mu <= 0 or n == 0:
        return float("-inf")
    return (-n * math.log(1.0 - math.exp(-mu))
            + math.log(mu) * sum(ys) - n * mu
            - sum(math.lgamma(v + 1.0) for v in ys))


def zero_truncated_poisson_mle(y_positive, tol=1e-12,
                               max_iter=200):
    """p.652: the estimate of mu solves dLL+/dmu = 0, which reduces to
    (sum_i Y_i+)/N+ = mu/(1 - exp(-mu)); solved here by bisection on
    that identity."""
    ys = _flat(y_positive)
    n = len(ys)
    if n == 0:
        raise ValueError("need at least one positive observation")
    target = sum(ys) / n
    if target <= 1.0:
        return 0.0
    lo, hi = 1e-9, 1.0
    while hi / (1.0 - math.exp(-hi)) < target:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        v = mid / (1.0 - math.exp(-mid))
        if abs(v - target) < tol:
            return mid
        if v < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def zap_best_split(y, x, candidates=None):
    """p.652: for a candidate split the zero-truncated log-likelihood
    (15.2) is computed separately in the two children and the best
    split maximizes LL+(left) + LL+(right)."""
    ys = _flat(y)
    xs = _flat(x)
    vals = sorted(set(xs)) if candidates is None else _flat(candidates)
    best = None
    for v in vals:
        L = [ys[i] for i in range(len(ys)) if xs[i] <= v and ys[i] > 0]
        R = [ys[i] for i in range(len(ys)) if xs[i] > v and ys[i] > 0]
        if not L or not R:
            continue
        ll = (zero_truncated_poisson_loglik(
                  L, zero_truncated_poisson_mle(L))
              + zero_truncated_poisson_loglik(
                  R, zero_truncated_poisson_mle(R)))
        if best is None or ll > best[1]:
            best = (v, ll)
    return {"threshold": best[0] if best else None,
            "loglik": best[1] if best else float("-inf")}


def zap_predict(theta_hat, mu_hat):
    """eq. (15.3) p.652: under ZAP_RF the prediction is the mean of
    the zero-altered Poisson,
    Y-hat = (1 - theta-hat) mu-hat / (1 - exp(-mu-hat)).

    Book erratum.  Equation (15.3) as printed, and the E(Y) line on
    p.651, both give the numerator as (1 - theta) exp(-mu), dropping
    the mu factor.  Three things on those same two pages show the mu
    belongs there:

    * the ZAP probability mass function printed directly above E(Y) is
      P(Y = y) = (1 - theta) exp(-mu) mu^y / ((1 - exp(-mu)) y!) for
      y > 0, whose mean is (1 - theta) mu / (1 - exp(-mu));
    * the Var(Y) expression printed on the very next line subtracts
      ((1 - theta) mu / (1 - exp(-mu)))^2, i.e. the square of the mean
      -- so the book's own variance formula uses the mu;
    * the estimating equation for mu on p.652,
      sum_i Y_i+ / N+ = mu / (1 - exp(-mu)), is the zero-truncated
      Poisson mean, again with mu in the numerator.

    The printed form is also not a count: it decreases towards zero as
    mu grows.  We implement the internally consistent formula."""
    th = float(theta_hat)
    mu = float(mu_hat)
    denom = 1.0 - math.exp(-mu)
    if denom <= 0:
        return 0.0
    return (1.0 - th) * mu / denom


def zap_mean_variance(theta, mu):
    """Mean and variance of the zero-altered Poisson, p.651.  The
    variance is transcribed exactly as printed; the mean is the
    corrected form documented in :func:`zap_predict`."""
    th = float(theta)
    mu = float(mu)
    denom = 1.0 - math.exp(-mu)
    if denom <= 0:
        return {"mean": 0.0, "variance": 0.0}
    k = (1.0 - th) / denom
    mean = k * mu
    return {"mean": mean, "variance": k * (mu + mu * mu) - mean * mean}


def zapc_predict(theta_hat, mu_hat, threshold=0.5):
    """eq. (15.4) p.652: under ZAPC_RF the prediction is 0 when
    theta-hat > 0.5 and mu-hat otherwise -- the probability is
    converted to a zero rather than to a binary label, and the 0.5
    threshold assumes no prior information."""
    return 0.0 if float(theta_hat) > float(threshold) \
        else float(mu_hat)


# --------- marginal structural models (Robins et al. 2000)
def msm_weighted_glm(y, X, weights=None, family="gaussian",
                     offset=None, n_iter=60, tol=1e-10):
    """Weighted GLM by iteratively reweighted least squares, the
    outcome stage of a marginal structural model.

    With the stabilized IPT weights of Robins, Hernan and Brumback
    (2000) as ``weights``, fitting E[Y | a-bar] in the pseudo-
    population created by the weights estimates the causal parameter
    of the MSM rather than an association adjusted for confounders.
    ``family`` is one of gaussian (identity), binomial (logit) or
    poisson (log).
    """
    Xm = _mat(X)
    ys = _flat(y)
    n = len(ys)
    p = len(Xm[0])
    w = [1.0] * n if weights is None else _flat(weights)
    off = [0.0] * n if offset is None else _flat(offset)
    beta = [0.0] * p
    if family == "gaussian":
        A = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
              for b in range(p)] for a in range(p)]
        rhs = [sum(w[i] * Xm[i][a] * (ys[i] - off[i])
                   for i in range(n)) for a in range(p)]
        beta = _solve(A, rhs)
        eta = [off[i] + sum(Xm[i][j] * beta[j] for j in range(p))
               for i in range(n)]
        mu = eta
    else:
        if family == "poisson":
            beta[0] = math.log(max(sum(w[i] * ys[i]
                                       for i in range(n))
                                   / max(sum(w), 1e-9), 1e-6))
        for _ in range(int(n_iter)):
            eta = [off[i] + sum(Xm[i][j] * beta[j]
                                for j in range(p))
                   for i in range(n)]
            if family == "binomial":
                mu = [1.0 / (1.0 + math.exp(-max(min(v, 700.0),
                                                 -700.0)))
                      for v in eta]
                Wd = [max(mu[i] * (1.0 - mu[i]), 1e-9)
                      for i in range(n)]
            elif family == "poisson":
                mu = [math.exp(min(v, 700.0)) for v in eta]
                Wd = [max(mu[i], 1e-9) for i in range(n)]
            else:
                raise ValueError("unknown family: %s" % family)
            z = [eta[i] - off[i] + (ys[i] - mu[i]) / Wd[i]
                 for i in range(n)]
            ww = [w[i] * Wd[i] for i in range(n)]
            A = [[sum(ww[i] * Xm[i][a] * Xm[i][b]
                      for i in range(n)) for b in range(p)]
                 for a in range(p)]
            rhs = [sum(ww[i] * Xm[i][a] * z[i] for i in range(n))
                   for a in range(p)]
            new = _solve(A, rhs)
            gap = max(abs(new[j] - beta[j]) for j in range(p))
            beta = new
            if gap < tol:
                break
        eta = [off[i] + sum(Xm[i][j] * beta[j] for j in range(p))
               for i in range(n)]
        mu = [1.0 / (1.0 + math.exp(-max(min(v, 700.0), -700.0)))
              if family == "binomial"
              else math.exp(min(v, 700.0)) for v in eta]
    return {"beta": beta, "fitted": mu, "eta": eta,
            "weights": w, "family": family}


def msm_design(treatment_history, extra=None):
    """The MSM design [1, a-bar] where a-bar is the cumulative
    treatment history, optionally with baseline covariates appended
    (only baseline covariates may enter an MSM; time-varying ones are
    handled through the weights)."""
    A = _mat(treatment_history)
    abar = [sum(row) for row in A]
    rows = [[1.0, abar[i]] for i in range(len(A))]
    if extra is not None:
        E = _mat(extra)
        rows = [rows[i] + list(E[i]) for i in range(len(rows))]
    return {"X": rows, "a_bar": abar}


def msm_cox_weighted(time, event, treatment_history, weights=None,
                     n_iter=60, tol=1e-10):
    """A weighted Cox marginal structural model: the partial
    likelihood is weighted by the stabilized IPT weights, so the
    hazard ratio it returns is marginal (causal) rather than
    covariate-conditional."""
    ts = _flat(time)
    ev = _flat(event)
    A = _mat(treatment_history)
    a = [sum(row) for row in A]
    n = len(ts)
    w = [1.0] * n if weights is None else _flat(weights)
    order = sorted(range(n), key=lambda i: ts[i])
    beta = 0.0
    for _ in range(int(n_iter)):
        g = h = 0.0
        for idx in order:
            if ev[idx] <= 0:
                continue
            risk = [j for j in range(n) if ts[j] >= ts[idx]]
            num = sum(w[j] * a[j] * math.exp(beta * a[j])
                      for j in risk)
            den = sum(w[j] * math.exp(beta * a[j]) for j in risk)
            num2 = sum(w[j] * a[j] * a[j] * math.exp(beta * a[j])
                       for j in risk)
            if den <= 0:
                continue
            g += w[idx] * (a[idx] - num / den)
            h += w[idx] * (num2 / den - (num / den) ** 2)
        if h <= 1e-12:
            break
        step = g / h
        beta += step
        if abs(step) < tol:
            break
    return {"beta": beta, "hazard_ratio": math.exp(beta)}


def msm_gmm(y, X, Z, weights=None, n_iter=60, tol=1e-10):
    """The GMM/estimating-equation form of an MSM: solve
    E[Z (Y - g(a-bar; beta))] = 0 with the IPT weights folded into the
    moment condition (Hansen 1982; Robins 1999).  With a linear g and
    as many instruments as parameters this is exactly identified and
    reduces to weighted two-stage least squares."""
    Xm = _mat(X)
    Zm = _mat(Z)
    ys = _flat(y)
    n = len(ys)
    w = [1.0] * n if weights is None else _flat(weights)
    ZtWX = [[sum(w[i] * Zm[i][a] * Xm[i][b] for i in range(n))
             for b in range(len(Xm[0]))] for a in range(len(Zm[0]))]
    ZtWy = [sum(w[i] * Zm[i][a] * ys[i] for i in range(n))
            for a in range(len(Zm[0]))]
    if len(Zm[0]) == len(Xm[0]):
        beta = _solve(ZtWX, ZtWy)
    else:                       # over-identified: two-step GMM
        A = _mm(_t(ZtWX), ZtWX)
        b = _mv(_t(ZtWX), ZtWy)
        beta = _solve(A, b)
    resid = [ys[i] - sum(Xm[i][j] * beta[j]
                         for j in range(len(beta)))
             for i in range(n)]
    moments = [sum(w[i] * Zm[i][a] * resid[i] for i in range(n)) / n
               for a in range(len(Zm[0]))]
    return {"beta": beta, "moments": moments, "residuals": resid}


# ---------------------------------------------------------------
# MVSML ch8 eq (8.13), ch9 eqs (9.1)-(9.47), ch14 eqs (14.1)-(14.14)
# ---------------------------------------------------------------


def khatri_rao_rows(A, B):
    """Row-wise Kronecker product, the ":" of eq. (8.13) p.296.

    Row i of the result is the outer product of row i of A with row i
    of B, flattened.  With A of order n x a and B of order n x b the
    result is n x (a*b).  The book writes P_u2 = P_u1 : Z_E and calls
    it the interaction between the two design matrices.
    """
    Am, Bm = _mat(A), _mat(B)
    return [[u * v for u in ra for v in rb] for ra, rb in zip(Am, Bm)]


def approx_kernel_extended(X, m_index, Z_u1, Z_E, kernel="linear",
                           gamma=None, tol=1e-10):
    """eq. (8.13) p.296: y = mu 1 + Z_E beta_E + P_u1 f + P_u2 l + eps.

    Steps 1-7 of the summary on p.296.  P = K_{L,m} U S^(-1/2) is the
    compressed design of (8.12) built from m of the L lines
    (sparse_kernel_design); Z_u1 expands it from lines to the n
    phenotypic records, P_u1 = Z_u1 P of order n x m; and
    P_u2 = P_u1 : Z_E of order n x mI is the row-wise Kronecker
    interaction with the environment design.  Step 8 then fits the
    stacked design under ridge, which is what the returned blocks are
    for.
    """
    sk = sparse_kernel_design(X, m_index, kernel=kernel, gamma=gamma,
                              tol=tol)
    P = sk["P"]
    Zu1 = _mat(Z_u1)
    ZE = _mat(Z_E)
    Pu1 = _mm(Zu1, P)
    Pu2 = khatri_rao_rows(Pu1, ZE)
    n = len(Pu1)
    design = [[1.0] + list(ZE[i]) + list(Pu1[i]) + list(Pu2[i])
              for i in range(n)]
    return {"P": P, "P_u1": Pu1, "P_u2": Pu2, "design": design,
            "widths": {"intercept": 1, "environments": len(ZE[0]),
                       "lines": len(Pu1[0]),
                       "line_x_env": len(Pu2[0])},
            "rank": sk["rank"]}


def hyperplane_value(X, beta0, beta):
    """eqs. (9.1) p.339 and (9.2) p.339: a hyperplane of a
    p-dimensional space is the (p-1)-dimensional flat subspace on
    which beta_0 + beta_1 x_1 + ... + beta_p x_p = 0.  (9.1) is the
    p = 3 case the book writes first.  Points whose left-hand side is
    < 0 satisfy (9.3) p.339 and lie on one side; those with it > 0
    satisfy (9.4) p.340 and lie on the other.  |f(x)| / ||beta|| is
    the Euclidean distance to the plane (p.345).
    """
    B = _flat(beta)
    nb = math.sqrt(sum(b * b for b in B))
    vals = [float(beta0) + sum(a * b for a, b in zip(row, B))
            for row in _mat(X)]
    return {"value": vals,
            "side": [1 if v > 0 else (-1 if v < 0 else 0)
                     for v in vals],
            "below": [v < 0 for v in vals],
            "above": [v > 0 for v in vals],
            "on_plane": [abs(v) <= 1e-12 for v in vals],
            "distance": [abs(v) / nb if nb > 0 else float("inf")
                         for v in vals],
            "norm_beta": nb}


def max_margin_classifier(X, y, **kw):
    """eq. (9.6) p.344 solved through its equivalent (9.7)-(9.8) p.346.

    (9.6) maximizes the margin M over beta subject to
    sum_j beta_j^2 = 1 and y_i(beta_0 + x_i beta) >= M.  p.345 shows
    M = 1 / ||beta|| once the scale is fixed by
    y_i(beta_0 + x_i beta) >= 1, so (9.6) is equivalent to minimizing
    (1/2)||beta||^2 subject to (9.8), which is (9.7).  The whole
    street is 2M = 2 / ||beta||.
    """
    fit = svm_fit_dual(X, y, C=None, **kw)
    beta, b0 = fit["beta"], fit["beta0"]
    nb = math.sqrt(sum(b * b for b in beta))
    ys = _flat(y)
    f = [b0 + sum(a * b for a, b in zip(row, beta))
         for row in _mat(X)]
    fm = [ys[i] * f[i] for i in range(len(ys))]
    return {"beta": beta, "beta0": b0, "norm_beta": nb,
            "margin": 1.0 / nb if nb > 0 else float("inf"),
            "street_width": 2.0 / nb if nb > 0 else float("inf"),
            "objective": 0.5 * nb * nb,
            "functional_margin": fm,
            "min_functional_margin": min(fm),
            "constraint_ok": min(fm) >= 1.0 - 1e-6,
            "alpha": fit["alpha"],
            "support_vectors": fit["support_vectors"]}


def wolfe_dual(f, grad_f, h=None, grad_h=None, g=None, grad_g=None,
               lam=None, alpha=None):
    """eqs. (9.9)-(9.14) pp.346-347, the general Wolfe dual.

    (9.9) minimizes f(x) over x in R^n subject to the m equalities
    h_i(x) = 0 of (9.10) and the p inequalities g_i(x) <= 0 of (9.11).
    (9.12) maximizes L = f(x) - sum_i lambda_i h_i(x)
    - sum_i alpha_i g_i(x) over (x, lambda, alpha), subject to the
    stationarity condition (9.13),
    grad f - sum_i lambda_i grad h_i - sum_i alpha_i grad g_i = 0, and
    to alpha_i >= 0 of (9.14).  All arguments are the numeric values
    of those functions and gradients at the point being checked, so
    this evaluates the dual objective and the two feasibility
    residuals without committing to any particular f.

    Sign convention: the book notes under (9.14) that the sign of the
    inequality term is crucial, and its own Illustrative Example 9.1
    writes the dual of "minimize x^2 subject to x >= 1" as
    x^2 - 2 alpha (x - 1).  The inequality is therefore supplied in
    the >= form, g_i(x) >= 0, and subtracted, exactly as (9.12) and
    (9.13) print it.  Note also that the two illustrative examples
    carry a factor 2 on the multiplier that the general (9.12) does
    not, so the alpha = 1 they report corresponds to alpha = 2 here.
    """
    hv = _flat(h) if h is not None else []
    gv = _flat(g) if g is not None else []
    lm = _flat(lam) if lam is not None else [0.0] * len(hv)
    al = _flat(alpha) if alpha is not None else [0.0] * len(gv)
    gf = _flat(grad_f)
    Gh = _mat(grad_h) if grad_h is not None else []
    Gg = _mat(grad_g) if grad_g is not None else []
    L = float(f) - sum(l * v for l, v in zip(lm, hv)) \
        - sum(a * v for a, v in zip(al, gv))
    stat = []
    for j in range(len(gf)):
        s = gf[j]
        for i in range(len(Gh)):
            s -= lm[i] * Gh[i][j]
        for i in range(len(Gg)):
            s -= al[i] * Gg[i][j]
        stat.append(s)
    return {"L": L, "stationarity": stat,
            "max_stationarity": max((abs(v) for v in stat),
                                    default=0.0),
            "alpha_nonnegative": all(a >= -1e-12 for a in al),
            "n_equality": len(hv), "n_inequality": len(gv)}


def qp_one_linear_constraint(a, c):
    """eqs. (9.15)-(9.26) pp.346-347, worked through the Wolfe dual.

    minimize z'z subject to a'z >= c.  The Wolfe dual of (9.17) p.347
    is L = z'z - 2 alpha (a'z - c); stationarity (9.18) gives
    z = alpha a, and substituting it back (9.19) leaves
    L(alpha) = -(a'a) alpha^2 + 2 c alpha, maximized at
    alpha = c / (a'a) >= 0 of (9.20).

    Illustrative Example 9.1 (9.15)-(9.20) is a = [1], c = 1, for
    which L(alpha) = -alpha^2 + 2 alpha and z = alpha = 1, exactly as
    printed.  Illustrative Example 9.2 (9.21)-(9.26) is a = [1, 1],
    c = 2, for which L(alpha) = -2 alpha^2 + 4 alpha and
    x = y = alpha = 1, also exactly as printed.  The two examples are
    the same problem, so one routine answers both.
    """
    av = _flat(a)
    aa = sum(v * v for v in av)
    if aa <= 0:
        raise ValueError("constraint vector a must be nonzero")
    alpha = float(c) / aa
    z = [alpha * v for v in av]
    return {"x": z, "alpha": alpha,
            "dual_quadratic": -aa, "dual_linear": 2.0 * float(c),
            "dual_value": -aa * alpha * alpha
                          + 2.0 * float(c) * alpha,
            "primal_value": sum(v * v for v in z),
            "constraint": sum(u * v for u, v in zip(av, z)),
            "active": True}


def svm_lagrangian(X, y, beta0, beta, alpha):
    """eq. (9.27) p.348: the Wolfe primal of the hard-margin problem,
    L(beta, beta_0, alpha) = (1/2)||beta||^2
    - sum_i alpha_i [ y_i(beta_0 + x_i beta) - 1 ],
    whose stationarity conditions are (9.28) and (9.29) and whose
    complementary slackness is (9.30).
    """
    B = _flat(beta)
    ys = _flat(y)
    Xm = _mat(X)
    f = [float(beta0) + sum(u * v for u, v in zip(row, B))
         for row in Xm]
    al = _flat(alpha)
    slack = [ys[i] * f[i] - 1.0 for i in range(len(ys))]
    return {"L": 0.5 * sum(b * b for b in B)
                 - sum(al[i] * slack[i] for i in range(len(al))),
            "quadratic_term": 0.5 * sum(b * b for b in B),
            "slack": slack,
            "grad_beta": [B[j] - sum(al[i] * ys[i] * Xm[i][j]
                                     for i in range(len(al)))
                          for j in range(len(B))],
            "grad_beta0": -sum(al[i] * ys[i]
                               for i in range(len(al)))}


def soft_margin_classifier(X, y, T, **kw):
    """eqs. (9.34)-(9.37) pp.354-355, the support vector classifier.

    (9.34) maximizes M over beta and the slacks; (9.35) fixes the
    scale with sum_j beta_j^2 = 1; (9.36) relaxes the margin
    constraint to y_i(beta_0 + sum_j beta_j x_ij) >= M(1 - zeta_i);
    and (9.37) is zeta_i >= 0 with sum_i zeta_i <= T.

    The book writes T both for that slack budget in (9.37) and for
    the box bound on the multipliers in (9.45); they are different
    parameters in the standard formulation and only (9.45) is
    directly solvable, so T here is the box bound of (9.45) and the
    realized sum of slacks is returned as slack_sum for comparison
    against a budget.  zeta_i is read off (9.36) as the hinge
    max(0, 1 - y_i(beta_0 + x_i beta)).
    """
    fit = svm_fit_dual(X, y, C=float(T), **kw)
    beta, b0 = fit["beta"], fit["beta0"]
    nb = math.sqrt(sum(b * b for b in beta))
    ys = _flat(y)
    f = [b0 + sum(u * v for u, v in zip(row, beta))
         for row in _mat(X)]
    zeta = [max(0.0, 1.0 - ys[i] * f[i]) for i in range(len(ys))]
    return {"beta": beta, "beta0": b0, "norm_beta": nb,
            "margin": 1.0 / nb if nb > 0 else float("inf"),
            "zeta": zeta, "slack_sum": sum(zeta),
            "n_violating": sum(1 for z in zeta if z > 1e-9),
            "n_misclassified": sum(1 for z in zeta if z > 1.0),
            "alpha": fit["alpha"],
            "support_vectors": fit["support_vectors"],
            "objective": fit["objective"]}


def soft_margin_kkt(X, y, beta0, beta, alpha, delta, zeta, T):
    """eqs. (9.38)-(9.43) pp.356-357, the Wolfe primal of the soft
    margin problem and its Karush-Kuhn-Tucker conditions.

    (9.38) L = (1/2)||beta||^2 + T sum_i zeta_i
    - sum_i alpha_i [ y_i(beta_0 + x_i beta) - 1 + zeta_i ]
    - sum_i delta_i zeta_i.

    The printed sign of the delta term on p.356 is inconsistent with
    the book's own (9.41), which states dL/dzeta_i = T - alpha_i
    - delta_i = 0; that requires the term to enter with a minus, and
    it is written with a minus here so that (9.41) holds.  The
    residuals returned are (9.39) beta - sum_i alpha_i y_i x_i,
    (9.40) sum_i alpha_i y_i, (9.41) alpha_i + delta_i - T,
    (9.42) alpha_i [ y_i(beta_0 + x_i beta) - 1 + zeta_i ], and
    (9.43) delta_i zeta_i.
    """
    B = _flat(beta)
    ys = _flat(y)
    Xm = _mat(X)
    al, dl, zt = _flat(alpha), _flat(delta), _flat(zeta)
    n = len(ys)
    f = [float(beta0) + sum(u * v for u, v in zip(row, B))
         for row in Xm]
    inner = [ys[i] * f[i] - 1.0 + zt[i] for i in range(n)]
    L = 0.5 * sum(b * b for b in B) + float(T) * sum(zt) \
        - sum(al[i] * inner[i] for i in range(n)) \
        - sum(dl[i] * zt[i] for i in range(n))
    r39 = [B[j] - sum(al[i] * ys[i] * Xm[i][j] for i in range(n))
           for j in range(len(B))]
    r40 = sum(al[i] * ys[i] for i in range(n))
    r41 = [al[i] + dl[i] - float(T) for i in range(n)]
    r42 = [al[i] * inner[i] for i in range(n)]
    r43 = [dl[i] * zt[i] for i in range(n)]
    worst = max([abs(v) for v in r39] + [abs(r40)]
                + [abs(v) for v in r41] + [abs(v) for v in r42]
                + [abs(v) for v in r43])
    return {"L": L, "stationarity_beta": r39, "balance": r40,
            "multiplier_sum": r41, "complementary_alpha": r42,
            "complementary_delta": r43, "max_residual": worst,
            "kkt_satisfied": worst < 1e-6}


def svm_soft_dual(X, y, T, K=None, **kw):
    """eqs. (9.44)-(9.45) p.357: the Wolfe dual of the support vector
    classifier, maximize L(alpha) = sum_i alpha_i
    - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j) subject to
    0 <= alpha_i <= T and sum_i alpha_i y_i = 0.  It differs from the
    hard-margin dual (9.32)-(9.33) only in the upper bound T on the
    multipliers, which is what the slack variables buy.
    """
    fit = svm_fit_dual(X, y, C=float(T), K=K, **kw)
    a = fit["alpha"]
    ys = _flat(y)
    return {"alpha": a, "beta": fit["beta"], "beta0": fit["beta0"],
            "objective": fit["objective"],
            "support_vectors": fit["support_vectors"],
            "balance": sum(a[i] * ys[i] for i in range(len(a))),
            "bounded": all(-1e-9 <= v <= float(T) + 1e-9 for v in a),
            "at_bound": [i for i in range(len(a))
                         if a[i] > float(T) - 1e-6]}


def ksvm_dual(X, y, T, kernel="linear", gamma=None, K=None, **kw):
    """eqs. (9.46)-(9.47) p.360: the support vector machine proper.
    Because the dual (9.44) touches the data only through the inner
    products x_i . x_j, every instance of one can be replaced by a
    positive definite symmetric kernel K(x_i, x_j), which is the
    kernel trick; the constraints (9.47) are unchanged from (9.45).
    The decision rule of p.360 is
    f(x) = sum_{i in S} alpha_i y_i K(x_i, x) + beta_0.
    """
    Km = _mat(K) if K is not None else \
        kernel_matrix(X, kernel=kernel, gamma=gamma)
    fit = svm_soft_dual(X, y, T, K=Km, **kw)
    fit["K"] = Km
    fit["kernel"] = kernel if K is None else "precomputed"
    return fit


def fda_integral(t, x_values, beta_values, mu=0.0):
    """eq. (14.1) p.579: the functional linear model with scalar
    response and one functional covariate,
    Y = mu + int_0^T x(t) beta(t) dt + E.  The integral of the
    product of the centered covariate curve and the coefficient
    function is evaluated by the trapezoid rule on the observation
    grid t, which is the same quadrature the chapter uses for the
    inner products of p.581.
    """
    ts = _flat(t)
    xs = _flat(x_values)
    bs = _flat(beta_values)
    s = 0.0
    for j in range(len(ts) - 1):
        dt = ts[j + 1] - ts[j]
        s += 0.5 * dt * (xs[j] * bs[j] + xs[j + 1] * bs[j + 1])
    return {"integral": s, "fitted": float(mu) + s,
            "mu": float(mu), "n_points": len(ts)}


def fda_basis_derivative(t, n_basis, p=1, kind="fourier",
                         period=None):
    """The p-th derivative of the basis functions of eq. (14.2) p.579,
    needed by the roughness penalty (14.11) p.601.  For the Fourier
    basis used by fda_basis_matrix, phi_0 = 1,
    phi_{2k-1}(t) = sin(2 pi k t / P) and
    phi_{2k}(t) = cos(2 pi k t / P), so the p-th derivative is
    (2 pi k / P)^p times a quarter-period phase shift; for the
    polynomial basis phi_l(t) = u^l with u = (t - lo)/span the p-th
    derivative is l!/(l-p)! u^(l-p) / span^p.
    """
    ts = _flat(t)
    L = int(n_basis)
    p = int(p)
    lo, hi = min(ts), max(ts)
    span = (hi - lo) or 1.0
    P = float(period) if period else span
    out = []
    for tv in ts:
        row = []
        for l in range(L):
            if kind == "fourier":
                if l == 0:
                    row.append(1.0 if p == 0 else 0.0)
                    continue
                k = (l + 1) // 2 if l % 2 == 1 else l // 2
                w = 2.0 * math.pi * k / P
                phase = w * tv + 0.5 * math.pi * p
                if l % 2 == 1:
                    row.append((w ** p) * math.sin(phase))
                else:
                    row.append((w ** p) * math.cos(phase))
            elif kind == "polynomial":
                if p > l:
                    row.append(0.0)
                else:
                    c = 1.0
                    for j in range(p):
                        c *= (l - j)
                    row.append(c * (((tv - lo) / span) ** (l - p))
                               / (span ** p))
            else:
                raise ValueError("unknown basis: %s" % kind)
        out.append(row)
    return out


def fda_penalty_matrix(t, L1, p=2, kind="fourier", period=None,
                       beta=None):
    """eq. (14.11) p.601: the roughness penalty
    J_beta = int_0^T [ d^p beta(t) / dt^p ]^2 dt.  With the basis
    expansion (14.2) the book writes J_beta = beta' P beta, where P is
    the L1 x L1 matrix with entries
    P_ij = int_0^T phi_i^(p)(t) phi_j^(p)(t) dt.  The integrals are
    taken by the trapezoid rule on the grid t.  Typical p is 1 or 2.
    """
    ts = _flat(t)
    D = fda_basis_derivative(ts, L1, p=p, kind=kind, period=period)
    m = len(ts)
    L = int(L1)
    P = [[0.0] * L for _ in range(L)]
    for i in range(L):
        for j in range(i, L):
            s = 0.0
            for q in range(m - 1):
                dt = ts[q + 1] - ts[q]
                s += 0.5 * dt * (D[q][i] * D[q][j]
                                 + D[q + 1][i] * D[q + 1][j])
            P[i][j] = s
            P[j][i] = s
    out = {"P": P, "order": p, "L1": L}
    if beta is not None:
        b = _flat(beta)
        out["J"] = sum(b[i] * P[i][j] * b[j]
                       for i in range(L) for j in range(L))
    return out


def fda_penalized_sse(y, X, beta, lam, P, mu=0.0):
    """eq. (14.10) p.599: the penalized sum of squared errors
    SSE_lambda(beta) = sum_i ( y_i - mu - sum_l x_il beta_l )^2
    + lambda J_beta, with J_beta = beta' P beta from (14.11).  lambda
    trades the fit of the first term against the smoothness of
    beta(); at lambda = 0 it is ordinary least squares and as lambda
    grows beta(t) is driven towards a constant.
    """
    ys = _flat(y)
    Xm = _mat(X)
    b = _flat(beta)
    Pm = _mat(P)
    fitted = [float(mu) + sum(u * v for u, v in zip(row, b))
              for row in Xm]
    resid = [ys[i] - fitted[i] for i in range(len(ys))]
    J = sum(b[i] * Pm[i][j] * b[j]
            for i in range(len(b)) for j in range(len(b)))
    sse = sum(r * r for r in resid)
    return {"sse": sse, "penalty": J, "lambda": float(lam),
            "objective": sse + float(lam) * J,
            "fitted": fitted, "residuals": resid}


def fda_penalized_fit(y, X, P, lam, mu=None, tol=1e-10):
    """eq. (14.12) p.601: with the spectral decomposition
    P = Gamma D Gamma' of the penalty matrix, X* = X Gamma and
    beta* = Gamma' beta, the penalized criterion becomes
    SSE_lambda(beta*) = ||y - 1_n mu - X* beta*||^2
    + lambda beta*' D beta*, whose minimizer is
    beta* = (X*'X* + lambda D)^-1 X*'(y - 1_n mu) and whose original
    coefficients are beta = Gamma beta*.  When P is rank deficient the
    zero eigenvalues contribute nothing to the penalty, which is the
    reduction to lambda beta_1*' D_1 beta_1* the book notes.  mu
    defaults to the mean of y, as the model centers the response.
    """
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    Pm = _mat(P)
    L = len(Pm)
    S = [[0.5 * (Pm[i][j] + Pm[j][i]) for j in range(L)]
         for i in range(L)]
    vals, vecs = np.linalg.eigh(np.marr(S))
    d = [float(v) for v in vals._flat()]
    G = [[float(v) for v in row] for row in vecs._tolist()] \
        if hasattr(vecs, "_tolist") else _mat(vecs)
    m = float(sum(ys) / n) if mu is None else float(mu)
    Xs = _mm(Xm, G)
    A = _mm(_t(Xs), Xs)
    for i in range(L):
        A[i][i] += float(lam) * d[i]
    rhs = _mv(_t(Xs), [v - m for v in ys])
    bstar = _solve(A, rhs)
    beta = _mv(G, bstar)
    fitted = [m + sum(u * v for u, v in zip(row, beta))
              for row in Xm]
    resid = [ys[i] - fitted[i] for i in range(n)]
    sse = sum(r * r for r in resid)
    pen = sum(float(lam) * d[i] * bstar[i] * bstar[i]
              for i in range(L))
    return {"beta": beta, "beta_star": bstar, "Gamma": G,
            "eigenvalues": d, "X_star": Xs, "mu": m,
            "fitted": fitted, "residuals": resid, "sse": sse,
            "penalty": pen, "objective": sse + pen,
            "rank": sum(1 for v in d if v > tol)}


def fda_env_interaction_design(X, env, reference=True):
    """The X_EF matrix printed on p.610 under eq. (14.14): the rows of
    X are laid out block-diagonally by environment, so record i in
    environment e contributes its functional scores in the columns
    belonging to e and zeros elsewhere, carrying the
    environment-by-reflectance interaction effects beta_EF.

    Written out for all I environments the blocks sum column by column
    to X exactly, so the joint design carrying both X and X_EF is rank
    deficient and beta and beta_EF are not separately identified by
    least squares.  The book fits (14.14) in BGLR, where the prior on
    each block resolves that.  The default reference=True drops the
    first environment block, leaving (I-1) L1 columns; that is the
    same reference coding the book applies to the environment design
    itself on p.607, where its code reads
    X_E = model.matrix(~0+Env, data = dat_F)[, -1], and it makes
    (14.14) identified.  Pass reference=False for the redundant
    parameterization exactly as printed.
    """
    Xm = _mat(X)
    e = list(env)
    levels = sorted(set(e), key=lambda v: (str(type(v)), v))
    keep = levels[1:] if reference else levels
    L = len(Xm[0])
    out = []
    for i, row in enumerate(Xm):
        block = [0.0] * (len(keep) * L)
        if e[i] in keep:
            k = keep.index(e[i])
            for j in range(L):
                block[k * L + j] = row[j]
        out.append(block)
    return {"X_EF": out, "levels": levels, "kept_levels": keep,
            "reference": reference, "n_columns": len(out[0])}


def fda_env_model(y, X, X_E, X_EF=None, lam=0.0, P=None):
    """eqs. (14.13) p.607 and (14.14) p.610: the functional regression
    with environment effects, y = 1_n mu + X_E beta_E + X beta + e,
    and its extension with the environment-by-reflectance interaction,
    y = 1_n mu + X_E beta_E + X beta + X_EF beta_EF + e.  X carries
    the L1 functional scores of (14.4)-(14.5), X_E is the design
    matrix of the environments and X_EF the block-diagonal design of
    p.610.  Passing X_EF = None gives (14.13), passing it gives
    (14.14); the two differ only by that block, which is why one
    routine covers both.  A penalty matrix P and lambda apply the
    roughness penalty of (14.11) to the functional block alone, as in
    (14.12); with lam = 0 the fit is ordinary least squares.
    """
    ys = _flat(y)
    Xm = _mat(X)
    XE = _mat(X_E)
    n = len(ys)
    blocks = [[1.0] for _ in range(n)]
    widths = {"intercept": 1, "environments": len(XE[0]),
              "functional": len(Xm[0])}
    D = [blocks[i] + list(XE[i]) + list(Xm[i]) for i in range(n)]
    if X_EF is not None:
        XF = _mat(X_EF)
        widths["env_x_functional"] = len(XF[0])
        D = [D[i] + list(XF[i]) for i in range(n)]
    k = len(D[0])
    A = _mm(_t(D), D)
    if P is not None and float(lam) != 0.0:
        Pm = _mat(P)
        off = 1 + len(XE[0])
        for i in range(len(Pm)):
            for j in range(len(Pm)):
                A[off + i][off + j] += float(lam) * Pm[i][j]
    coef = _solve(A, _mv(_t(D), ys))
    fitted = _mv(D, coef)
    resid = [ys[i] - fitted[i] for i in range(n)]
    off = 1
    beta_E = coef[off:off + widths["environments"]]
    off += widths["environments"]
    beta = coef[off:off + widths["functional"]]
    off += widths["functional"]
    beta_EF = coef[off:] if X_EF is not None else []
    return {"coef": coef, "mu": coef[0], "beta_E": beta_E,
            "beta": beta, "beta_EF": beta_EF, "widths": widths,
            "design": D, "fitted": fitted, "residuals": resid,
            "sse": sum(r * r for r in resid), "n_columns": k,
            "has_interaction": X_EF is not None}
