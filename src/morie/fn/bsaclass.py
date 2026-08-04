# morie.fn -- bsaclass (rootcoder007/morie)
"""Pattern classification and decomposition: discriminants, Bayes, SVM, k-NN, clustering, PCA/ICA/NMF, sparse coding, validation and performance measures.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 56
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from fractions import Fraction
from math import cos, isfinite, sin, tanh
from math import erf, exp, fsum, lgamma as _lgamma, log, pi, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._richresult import RichResult

__all__ = [
    'accuracy',
    'rangayyan_accuracy',
    'mlpbp',
    'rangayyan_ann_mlp',
    'bayescls',
    'rangayyan_bayes_classifier',
    'bayesnorm',
    'rangayyan_bayes_gaussian',
    'bbb',
    'rangayyan_bundle_branch_block',
    'pvcbayes',
    'rangayyan_ecg_bbb_normal',
    'bcichsel',
    'rangayyan_bci_nmf',
    'normdist',
    'divergence',
    'divav',
    'kld',
    'pdfoverlap',
    'chernoff',
    'hellinger',
    'gaussoverlap',
    'rangayyan_bhattacharyya',
    'bpursuit',
    'rangayyan_basis_pursuit',
    'cadpipe',
    'rangayyan_cad_pipeline',
    'cnnsig',
    'rangayyan_cnn_signal',
    'fecgnmf',
    'rangayyan_fetal_ecg_single',
    'pvclindf',
    'rangayyan_ecg_normal_ectopic',
    'eegbands',
    'rangayyan_eeg_rhythms',
    'elbow',
    'rangayyan_kmeans_elbow',
    'seizdict',
    'rangayyan_epilepsy_ksvd',
    'errbound',
    'rangayyan_bayes_error_bound',
    'fishcrit',
    'rangayyan_fisher_criterion',
    'fishlda',
    'rangayyan_fisher_lda',
    'hclust',
    'rangayyan_hierarchical_clust',
    'icafix',
    'rangayyan_fastica',
    'icaclean',
    'rangayyan_ica_artifact',
    'infomax',
    'rangayyan_infomax_ica',
    'kfoldcv',
    'rangayyan_kfold_cv',
    'kmeans',
    'rangayyan_kmeans',
    'vagclass',
    'rangayyan_knee_classify',
    'knn',
    'rangayyan_knn_classifier',
    'ksvdfit',
    'rangayyan_ksvd',
    'dictcode',
    'rangayyan_dictionary_sparse',
    'lindisc',
    'rangayyan_linear_discrim',
    'lindsep',
    'rangayyan_lin_discr_sep',
    'loocv',
    'rangayyan_loo_cv',
    'logreg',
    'rangayyan_logistic_regression',
    'lstm',
    'rangayyan_lstm_signal',
    'mahal',
    'rangayyan_mahalanobis',
    'mcnemar',
    'rangayyan_mcnemar_test',
    'mpursuit',
    'rangayyan_matching_pursuit',
    'bmidec',
    'rangayyan_neural_decode',
    'nmfmu',
    'rangayyan_nmf',
    'nmfchsel',
    'rangayyan_nmf_channel_sel',
    'ompfit',
    'rangayyan_omp',
    'pcasig',
    'rangayyan_pca_signals',
    'mixcmp',
    'rangayyan_pca_vs_ica',
    'ppv',
    'rangayyan_ppv',
    'qda',
    'rangayyan_qda',
    'rbfn',
    'rangayyan_rbf_network',
    'roc',
    'rangayyan_roc_curve',
    'ahi',
    'rangayyan_sleep_apnea_nmf',
    'sens',
    'rangayyan_sensitivity',
    'sepindex',
    'rangayyan_separability_index',
    'spec',
    'rangayyan_specificity',
    'sparsecode',
    'rangayyan_sparse_rep',
    'svm',
    'rangayyan_svm',
    'svmkern',
    'rangayyan_svm_kernel',
    'vagtfd',
    'rangayyan_vag_adaptive_tfd',
    'rangayyan_ch4_pan_tompkins_peak_classification',
    'rangayyanksvd',
    'rangayyanloocv',
    'rangayyannmf',
    'rangayyanomp',
    'rangayyanppv',
    'rangayyanqda',
    'rangayyansvm',
]

# ---------------------------------------------------------------- shared arithmetic
# Small dense-linear-algebra and spectral helpers used by the adaptive-decomposition
# and classifier blocks below.  Pure standard library on purpose: morie.fn carries no
# external numeric dependency.

def _bxvec(v, name="x"):
    """Coerce to a non-empty list of finite floats."""
    out = aslist(v)
    if not out:
        raise ValueError(name + " must be a non-empty sequence")
    for t in out:
        if not isfinite(t):
            raise ValueError(name + " must contain only finite values")
    return out


def _bxmat(M, name="X"):
    """Coerce to a non-empty rectangular list-of-lists of finite floats."""
    if M is None:
        raise ValueError(name + " is required")
    try:
        rows = list(M)
    except TypeError:
        raise ValueError(name + " must be a sequence of rows")
    if not rows:
        raise ValueError(name + " must have at least one row")
    out = []
    for r in rows:
        rr = aslist(r)
        if not rr:
            raise ValueError(name + " rows must be non-empty")
        for t in rr:
            if not isfinite(t):
                raise ValueError(name + " must contain only finite values")
        out.append(rr)
    w = len(out[0])
    if any(len(r) != w for r in out):
        raise ValueError(name + " must be rectangular")
    return out


def _bxdot(a, b):
    return fsum(a[i] * b[i] for i in range(len(a)))


def _bxnrm(a):
    return sqrt(fsum(t * t for t in a))


def _bxtr(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _bxmm(A, B):
    if len(A[0]) != len(B):
        raise ValueError("inner matrix dimensions do not agree")
    Bt = _bxtr(B)
    return [[_bxdot(r, c) for c in Bt] for r in A]


def _bxmv(A, v):
    if len(A[0]) != len(v):
        raise ValueError("matrix and vector dimensions do not agree")
    return [_bxdot(r, v) for r in A]


def _bxmean(v):
    return fsum(v) / len(v)


def _bxsd(v, ddof=1):
    n = len(v)
    if n - ddof < 1:
        return 0.0
    m = _bxmean(v)
    return sqrt(fsum((t - m) ** 2 for t in v) / (n - ddof))


def _bxkurt(v):
    """Kurtosis excess K' = K - 3, Rangayyan eq. (3.5) and the note below it.

    Zero for a Gaussian; positive for a peaked, heavy-tailed PDF.
    """
    n = len(v)
    if n < 4:
        raise ValueError("kurtosis needs at least four samples")
    m = _bxmean(v)
    s2 = fsum((t - m) ** 2 for t in v) / n
    if s2 <= 0.0:
        return 0.0
    m4 = fsum((t - m) ** 4 for t in v) / n
    return m4 / (s2 * s2) - 3.0


def _bxsolve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    if n != len(b) or any(len(r) != n for r in A):
        raise ValueError("linear system is not square or is inconsistent")
    M = [list(A[i]) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("linear system is singular")
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / pv
            if f == 0.0:
                continue
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _bxlstsq(A, y, ridge=1e-10):
    """Ridge-regularised least squares via the normal equations."""
    At = _bxtr(A)
    G = _bxmm(At, A)
    for i in range(len(G)):
        G[i][i] += ridge
    return _bxsolve(G, _bxmv(At, y))


def _bxjacobi(S, sweeps=60, tol=1e-12):
    """Eigenpairs of a real symmetric matrix by the cyclic Jacobi rotation method.

    Chosen over power iteration because it returns the whole spectrum,
    including repeated eigenvalues, without deflation error.  Returns
    (values, vectors) with vectors as columns, sorted by decreasing value.
    """
    n = len(S)
    A = [list(r) for r in S]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = fsum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                c = 1.0 / sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    vals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    vecs = [[V[i][j] for j in order] for i in range(n)]
    return [vals[i] for i in order], vecs


def _bxrng(seed):
    """Deterministic LCG (Numerical Recipes ranqd1 constants) on (0, 1)."""
    st = [int(seed) & 0xFFFFFFFF]

    def nxt():
        st[0] = (1664525 * st[0] + 1013904223) & 0xFFFFFFFF
        return (st[0] + 0.5) / 4294967296.0

    return nxt


def _bxcov(X, unbiased=True):
    """Sample covariance of the columns of X (rows are observations)."""
    n, p = len(X), len(X[0])
    if unbiased and n < 2:
        raise ValueError("covariance needs at least two observations")
    mu = [fsum(X[i][j] for i in range(n)) / n for j in range(p)]
    d = n - 1 if unbiased else n
    C = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for b in range(a, p):
            v = fsum((X[i][a] - mu[a]) * (X[i][b] - mu[b]) for i in range(n)) / d
            C[a][b] = v
            C[b][a] = v
    return mu, C


def _bxnmfmu(V, r, maxiter, tol, seed, cost):
    """Lee-Seung multiplicative-update NMF core.

    ``cost="ls"``  -> squared-error updates, Rangayyan eqs. (9.49) and (9.50).
    ``cost="kld"`` -> divergence updates, Rangayyan eqs. (9.54) and (9.55).
    """
    m, n = len(V), len(V[0])
    for row in V:
        for t in row:
            if t < 0.0:
                raise ValueError("NMF requires a nonnegative matrix V")
    r = int(r)
    if r < 1 or r > min(m, n):
        raise ValueError("rank r must satisfy 1 <= r <= min(rows, cols) of V")
    u = _bxrng(seed)
    scale = sqrt(max(fsum(fsum(row) for row in V) / (m * n), 1e-12) / r)
    W = [[scale * (0.5 + u()) for _ in range(r)] for _ in range(m)]
    H = [[scale * (0.5 + u()) for _ in range(n)] for _ in range(r)]
    eps = 1e-12
    prev = None
    err = float("nan")
    it = 0
    for it in range(1, int(maxiter) + 1):
        if cost == "ls":
            Wt = _bxtr(W)
            WtV = _bxmm(Wt, V)
            WtWH = _bxmm(_bxmm(Wt, W), H)
            for a in range(r):
                for b in range(n):
                    H[a][b] *= WtV[a][b] / (WtWH[a][b] + eps)
            Ht = _bxtr(H)
            VHt = _bxmm(V, Ht)
            WHHt = _bxmm(W, _bxmm(H, Ht))
            for a in range(m):
                for b in range(r):
                    W[a][b] *= VHt[a][b] / (WHHt[a][b] + eps)
        else:
            R = _bxmm(W, H)
            Q = [[V[i][j] / (R[i][j] + eps) for j in range(n)] for i in range(m)]
            Ht = _bxtr(H)
            num = _bxmm(Q, Ht)
            for a in range(m):
                for b in range(r):
                    den = fsum(H[b][j] for j in range(n))
                    W[a][b] *= num[a][b] / (den + eps)
            R = _bxmm(W, H)
            Q = [[V[i][j] / (R[i][j] + eps) for j in range(n)] for i in range(m)]
            Wt = _bxtr(W)
            num = _bxmm(Wt, Q)
            for a in range(r):
                for b in range(n):
                    den = fsum(W[i][a] for i in range(m))
                    H[a][b] *= num[a][b] / (den + eps)
        R = _bxmm(W, H)
        err = sqrt(fsum((V[i][j] - R[i][j]) ** 2 for i in range(m) for j in range(n)))
        if prev is not None and abs(prev - err) <= tol * max(1.0, prev):
            break
        prev = err
    return W, H, err, it


def _bxomp(x, D, sparsity, tol):
    """Orthogonal matching pursuit: greedy atom picks, least squares on the support.

    D is a list of atoms (each a list the same length as x).  Returns
    (coefficients over the full dictionary, support in selection order,
    residual vector).
    """
    n = len(x)
    for a in D:
        if len(a) != n:
            raise ValueError("every dictionary atom must have the same length as x")
    norms = [_bxnrm(a) for a in D]
    if any(t <= 0.0 for t in norms):
        raise ValueError("dictionary atoms must have nonzero norm")
    r = list(x)
    sup = []
    coef = [0.0] * len(D)
    k = len(D) if sparsity is None else int(sparsity)
    if k < 1:
        raise ValueError("sparsity must be a positive integer")
    for _ in range(min(k, len(D), n)):
        if _bxnrm(r) <= tol:
            break
        best, bv = -1, -1.0
        for j in range(len(D)):
            if j in sup:
                continue
            v = abs(_bxdot(D[j], r)) / norms[j]
            if v > bv:
                best, bv = j, v
        if best < 0:
            break
        sup.append(best)
        A = [[D[j][i] for j in sup] for i in range(n)]
        w = _bxlstsq(A, list(x))
        coef = [0.0] * len(D)
        for idx, j in enumerate(sup):
            coef[j] = w[idx]
        approx = _bxmv(A, w)
        r = [x[i] - approx[i] for i in range(n)]
    return coef, sup, r


def _bxgabor(n, natoms, seed=1):
    """A real Gabor dictionary, Rangayyan eqs. (9.2) and (9.3).

    Atoms g_gamma(t) = s^-1/2 g((t - tau)/s) cos(2 pi f t + phi) with the
    Gaussian window g(t) = 2^(1/4) exp(-pi t^2).  Scales, translations and
    modulations are laid out on a fixed dyadic grid so the dictionary is
    reproducible without an RNG.
    """
    natoms = int(natoms)
    if n < 4 or natoms < 1:
        raise ValueError("need n >= 4 samples and at least one atom")
    atoms, params = [], []
    scales = [n / (2.0 ** j) for j in range(1, 6)]
    j = 0
    while len(atoms) < natoms:
        s = scales[j % len(scales)]
        step = max(1, int(s / 2.0))
        for tau in range(0, n, step):
            for f in (0.0, 0.5 / s, 1.0 / s, 2.0 / s, 4.0 / s):
                a = []
                for t in range(n):
                    z = (t - tau) / s
                    if abs(z) > 6.0:
                        a.append(0.0)
                    else:
                        a.append(2.0 ** 0.25 / sqrt(s) * exp(-pi * z * z)
                                 * cos(2.0 * pi * f * t))
                nr = _bxnrm(a)
                if nr <= 1e-12:
                    continue
                atoms.append([t / nr for t in a])
                params.append({"scale": s, "translation": float(tau), "frequency": f})
                if len(atoms) >= natoms:
                    return atoms, params
        j += 1
        if j > 64:
            break
    return atoms, params


def _bxdftmag(x):
    """Magnitude spectrum for k = 0 .. N//2 by direct evaluation.

    ponytail: O(N^2) DFT; swap in a radix-2 FFT if N ever exceeds a few thousand.
    """
    n = len(x)
    out = []
    for k in range(n // 2 + 1):
        w = -2.0 * pi * k / n
        re = fsum(x[t] * cos(w * t) for t in range(n))
        im = fsum(x[t] * sin(w * t) for t in range(n))
        out.append(sqrt(re * re + im * im))
    return out


def _bxstft(x, nwin, hop):
    """Hann-windowed STFT returning (real, imag, magnitude) frame lists."""
    n = len(x)
    if nwin < 4 or nwin > n:
        raise ValueError("window length must satisfy 4 <= nwin <= len(x)")
    if hop < 1 or hop > nwin:
        raise ValueError("hop must satisfy 1 <= hop <= nwin")
    win = [0.5 - 0.5 * cos(2.0 * pi * i / nwin) for i in range(nwin)]
    nb = nwin // 2 + 1
    re_f, im_f, mag_f = [], [], []
    st = 0
    while st + nwin <= n:
        seg = [x[st + i] * win[i] for i in range(nwin)]
        re_r, im_r, mg_r = [], [], []
        for k in range(nb):
            w = -2.0 * pi * k / nwin
            re = fsum(seg[t] * cos(w * t) for t in range(nwin))
            im = fsum(seg[t] * sin(w * t) for t in range(nwin))
            re_r.append(re)
            im_r.append(im)
            mg_r.append(sqrt(re * re + im * im))
        re_f.append(re_r)
        im_f.append(im_r)
        mag_f.append(mg_r)
        st += hop
    if not mag_f:
        raise ValueError("signal is too short for the requested window")
    return re_f, im_f, mag_f, win


def _bxistft(re_f, im_f, nwin, hop, win, n):
    """Overlap-add inverse of _bxstft (Hermitian completion of the half spectrum)."""
    out = [0.0] * n
    wsum = [0.0] * n
    nb = nwin // 2 + 1
    for fi in range(len(re_f)):
        st = fi * hop
        seg = [0.0] * nwin
        for t in range(nwin):
            acc = re_f[fi][0]
            for k in range(1, nb):
                w = 2.0 * pi * k * t / nwin
                c = 2.0 if (k < nwin - k) else 1.0
                acc += c * (re_f[fi][k] * cos(w) - im_f[fi][k] * sin(w))
            seg[t] = acc / nwin
        for t in range(nwin):
            if st + t < n:
                out[st + t] += seg[t] * win[t]
                wsum[st + t] += win[t] * win[t]
    return [out[i] / wsum[i] if wsum[i] > 1e-12 else 0.0 for i in range(n)]


def _bxconfusion(true, pred):
    """2 x 2 counts (TP, TN, FP, FN) for a binary problem coded 1 / 0."""
    tp = sum(1 for a, b in zip(true, pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(true, pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(true, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(true, pred) if a == 1 and b == 0)
    return tp, tn, fp, fn


def _bxscores(tp, tn, fp, fn):
    """Sensitivity, specificity and accuracy, Rangayyan eqs. (10.100), (10.101), (10.103)."""
    sen = tp / (tp + fn) if (tp + fn) else float("nan")
    spe = tn / (tn + fp) if (tn + fp) else float("nan")
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else float("nan")
    return sen, spe, acc

def _solve_lin(A, b):
    """Solve A x = b by Gauss-Jordan with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("the system is singular")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _tanh(v):
    """tanh without importing it, stable for large |v|."""
    if v > 20.0:
        return 1.0
    if v < -20.0:
        return -1.0
    e = exp(2.0 * v)
    return (e - 1.0) / (e + 1.0)

def _mat(m):
    """Accept a matrix as a list of rows and return a list of lists."""
    return [aslist(r) for r in m]


def _colmeans(X):
    n = len(X)
    p = len(X[0])
    return [fsum(row[j] for row in X) / n for j in range(p)]


def _scatter(X, mu):
    """Sum of (x - mu)(x - mu)^T over the rows -- the SCATTER, not the
    covariance: it is not divided by n."""
    p = len(mu)
    S = [[0.0] * p for _ in range(p)]
    for row in X:
        d = [row[j] - mu[j] for j in range(p)]
        for i in range(p):
            for j in range(p):
                S[i][j] += d[i] * d[j]
    return S


def _inv(M):
    """Gauss-Jordan inverse, raising rather than returning garbage."""
    n = len(M)
    A = [list(M[i]) + [1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-300:
            raise ValueError("the matrix is singular and cannot be "
                             "inverted; a class may have fewer samples "
                             "than features")
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [v / piv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [A[r][k] - f * A[c][k] for k in range(2 * n)]
    return [row[n:] for row in A]


def _det(M):
    """Determinant by LU with partial pivoting."""
    n = len(M)
    A = [list(r) for r in M]
    d = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-300:
            return 0.0
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= f * A[c][k]
    return d


def _trace(M):
    return fsum(M[i][i] for i in range(len(M)))


def _matmul(A, B):
    return [[fsum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def _lower_gamma(s, x):
    """Regularized lower incomplete gamma P(s, x), series then continued
    fraction -- enough for the chi-square tail."""
    if x < 0 or s <= 0:
        raise ValueError("the incomplete gamma needs s > 0 and x >= 0")
    if x == 0:
        return 0.0
    if x < s + 1.0:
        term = 1.0 / s
        total = term
        n = s
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * exp(-x + s * log(x) - _lgamma(s))
    # Lentz's continued fraction for the upper tail
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - exp(-x + s * log(x) - _lgamma(s)) * h


def _chisq_sf(stat, df):
    """Upper tail of the chi-square distribution."""
    if stat <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - _lower_gamma(df / 2.0, stat / 2.0)))


def _groups(X, y):
    """Split rows of X by label, preserving first-seen label order."""
    order, out = [], {}
    for row, lab in zip(X, y):
        if lab not in out:
            out[lab] = []
            order.append(lab)
        out[lab].append(row)
    return order, out


# -- rgacc: Classification accuracy.
def accuracy(table=None, tp=None, tn=None, fp=None, fn=None,
             prevalence=None, kind=None, exact=False):
    """Classification accuracy -- every definition, not just one.

    The book gives two.  Eq. (10.102), stated first, is prevalence
    weighted:

        accuracy = S+ P(A) + S- P(N)

    and eq. (10.103) is the fallback used only "if the prior
    probabilities are not available":

        accuracy = (TP + TN) / (TP + TN + FP + FN)

    Eq. (10.103) IS eq. (10.102) evaluated at the prevalence of the TEST
    SET, so on a set deliberately balanced 50/50 it describes a
    population that does not exist.  A third form is in common use and
    is included because it is what "balanced accuracy" means elsewhere:
    the unweighted mean of sensitivity and specificity, which is
    eq. (10.102) at a prevalence of one half.

    ``kind`` selects the headline value -- "raw", "weighted" or
    "balanced" -- and every form is returned regardless, so no caller has
    to recompute one from another.  Supplying ``prevalence`` selects
    "weighted" unless ``kind`` says otherwise.

    ``exact`` returns Fractions instead of floats.  The inputs are
    integer counts, so the raw accuracy is a RATIO OF INTEGERS and is
    representable exactly; rounding it to a float is a choice, not a
    necessity.  With an exact prevalence (an int, a Fraction, or a
    decimal string) the weighted form stays exact too.
    """
    if table is not None:
        t = [aslist(r) for r in table]
        if len(t) != 2 or any(len(r) != 2 for r in t):
            raise ValueError("the table must be 2x2, "
                             "[[TP, FN], [FP, TN]]")
        TP, FN, FP, TN = t[0][0], t[0][1], t[1][0], t[1][1]
    else:
        if None in (tp, tn, fp, fn):
            raise ValueError("give a 2x2 table or all four of tp, tn, "
                             "fp, fn")
        TP, TN, FP, FN = tp, tn, fp, fn
    counts = [TP, TN, FP, FN]
    if any(float(v) < 0 for v in counts):
        raise ValueError("counts cannot be negative")
    if any(float(v) != int(float(v)) for v in counts):
        raise ValueError("counts must be whole numbers")
    TP, TN, FP, FN = (int(float(v)) for v in counts)
    total = TP + TN + FP + FN
    if total <= 0:
        raise ValueError("the table is empty")
    if TP + FN <= 0 or TN + FP <= 0:
        raise ValueError("a class is empty; the sensitivity or "
                         "specificity is undefined")
    kinds = ("raw", "weighted", "balanced")
    if kind is not None and kind not in kinds:
        raise ValueError("kind must be one of %s, got %r"
                         % (", ".join(kinds), kind))

    num = Fraction if exact else (lambda a, b=1: float(a) / float(b))
    se = Fraction(TP, TP + FN) if exact else TP / (TP + FN)
    sp = Fraction(TN, TN + FP) if exact else TN / (TN + FP)
    raw = Fraction(TP + TN, total) if exact else (TP + TN) / total
    test_prev = Fraction(TP + FN, total) if exact else (TP + FN) / total
    half = Fraction(1, 2) if exact else 0.5
    balanced = half * (se + sp)

    weighted = None
    prev = None
    if prevalence is not None:
        prev = Fraction(str(prevalence)) if exact else float(prevalence)
        if not 0 <= prev <= 1:
            raise ValueError("the prevalence must lie in [0, 1]")
        weighted = se * prev + sp * (1 - prev)

    chosen = kind
    if chosen is None:
        chosen = "weighted" if prevalence is not None else "raw"
    if chosen == "weighted" and weighted is None:
        raise ValueError("kind='weighted' needs a prevalence; without "
                         "the priors the book falls back on "
                         "eq. (10.103), kind='raw'")
    headline = {"raw": raw, "weighted": weighted,
                "balanced": balanced}[chosen]

    return RichResult(payload={
        "accuracy": headline, "kind": chosen,
        "raw_accuracy": raw, "weighted_accuracy": weighted,
        "balanced_accuracy": balanced,
        "sensitivity": se, "specificity": sp,
        "prevalence": prev, "test_set_prevalence": test_prev,
        "counts": {"tp": TP, "tn": TN, "fp": FP, "fn": FN},
        "n": total, "exact": bool(exact),
        "prior_weighted": chosen == "weighted",
        "balanced_is_eq_10_102_at_one_half": True,
        "eq_10_103_is_eq_10_102_at_the_test_set_prevalence": True,
        "method": "Rangayyan (2024) eqs. (10.102)-(10.103), with the "
                  "balanced form at P(A) = 1/2"})


rangayyan_accuracy = accuracy  # pre-policy spelling


# -- rgann: Multilayer perceptron (ANN) with backpropagation.
def mlpbp(X, y, hidden=4, eta=0.5, alpha=0.9, maxiter=500, tol=1e-4, seed=1):
    """Train a two-layer perceptron by back-propagation.

    Why: when neither the prior probabilities nor any symbolic rule base is
    available for a diagnostic problem, a network that infers the decision
    surface directly from labelled instances is the practical alternative to
    the parametric classifiers.  Rangayyan, *Biomedical Signal Analysis*, 3rd
    ed., Section 10.8 (Figure 10.5).

    Forward pass, eqs. (10.79), (10.80) and the logistic node function (10.81):

        x#_j = f(sum_i w_ij x_i - theta_j),
        y_k  = f(sum_j w#_jk x#_j - theta#_k),   f(b) = 1 / (1 + exp(-b)).

    Weights and offsets are updated by eqs. (10.82) to (10.85), each carrying
    the gain term ``eta`` and the momentum term ``alpha`` of the book.

    Parameters
    ----------
    X : sequence of sequences
        One pattern vector per row.
    y : sequence
        Desired output per pattern; 0/1 for two classes, or an integer class
        index, in which case one output node per class is used.
    hidden : int
        Number of hidden nodes J.
    eta, alpha : float
        Gain and momentum terms of eqs. (10.82) to (10.85).
    maxiter : int
        Maximum number of passes over the training set.
    tol : float
        Stop once the mean squared output error falls below this.
    seed : int
        Seed of the deterministic generator used for the initial weights.

    Returns
    -------
    RichResult
        Keys ``weights``, ``offsets``, ``predictions``, ``classes``,
        ``accuracy``, ``mse``, ``iterations``, ``method``.
    """
    X = _bxmat(X, "X")
    y = _bxvec(y, "y")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    hidden = int(hidden)
    if hidden < 1:
        raise ValueError("hidden must be a positive integer")
    if not (0.0 < eta <= 10.0) or not (0.0 <= alpha < 1.0):
        raise ValueError("need 0 < eta <= 10 and 0 <= alpha < 1")
    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    labels = sorted(set(int(t) for t in y))
    if len(labels) < 2:
        raise ValueError("y must contain at least two distinct classes")
    K = 1 if len(labels) == 2 else len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    D = []
    for t in y:
        if K == 1:
            D.append([float(idx[int(t)])])
        else:
            D.append([1.0 if i == idx[int(t)] else 0.0 for i in range(K)])

    n, I = len(X), len(X[0])
    u = _bxrng(seed)
    W1 = [[u() - 0.5 for _ in range(hidden)] for _ in range(I)]
    T1 = [u() - 0.5 for _ in range(hidden)]
    W2 = [[u() - 0.5 for _ in range(K)] for _ in range(hidden)]
    T2 = [u() - 0.5 for _ in range(K)]
    dW1 = [[0.0] * hidden for _ in range(I)]
    dT1 = [0.0] * hidden
    dW2 = [[0.0] * K for _ in range(hidden)]
    dT2 = [0.0] * K

    def sig(b):
        if b < -700.0:
            return 0.0
        if b > 700.0:
            return 1.0
        return 1.0 / (1.0 + exp(-b))

    mse = float("nan")
    it = 0
    for it in range(1, maxiter + 1):
        tot = 0.0
        for s in range(n):
            xs = X[s]
            xh = [sig(fsum(W1[i][j] * xs[i] for i in range(I)) - T1[j])
                  for j in range(hidden)]
            yo = [sig(fsum(W2[j][k] * xh[j] for j in range(hidden)) - T2[k])
                  for k in range(K)]
            dk = [yo[k] * (1.0 - yo[k]) * (D[s][k] - yo[k]) for k in range(K)]
            tot += fsum((D[s][k] - yo[k]) ** 2 for k in range(K))
            for j in range(hidden):
                for k in range(K):
                    step = eta * dk[k] * xh[j] + alpha * dW2[j][k]
                    W2[j][k] += step
                    dW2[j][k] = step
            for k in range(K):
                step = -eta * dk[k] + alpha * dT2[k]
                T2[k] += step
                dT2[k] = step
            bp = [xh[j] * (1.0 - xh[j]) * fsum(dk[k] * W2[j][k] for k in range(K))
                  for j in range(hidden)]
            for i in range(I):
                for j in range(hidden):
                    step = eta * bp[j] * xs[i] + alpha * dW1[i][j]
                    W1[i][j] += step
                    dW1[i][j] = step
            for j in range(hidden):
                step = -eta * bp[j] + alpha * dT1[j]
                T1[j] += step
                dT1[j] = step
        mse = tot / (n * K)
        if mse <= tol:
            break

    pred, raw = [], []
    for s in range(n):
        xs = X[s]
        xh = [sig(fsum(W1[i][j] * xs[i] for i in range(I)) - T1[j])
              for j in range(hidden)]
        yo = [sig(fsum(W2[j][k] * xh[j] for j in range(hidden)) - T2[k])
              for k in range(K)]
        raw.append(yo)
        if K == 1:
            pred.append(labels[1] if yo[0] >= 0.5 else labels[0])
        else:
            pred.append(labels[max(range(K), key=lambda k: yo[k])])
    acc = fsum(1.0 for a, b in zip(y, pred) if int(a) == b) / n
    return RichResult(payload={
        "weights": {"input_hidden": W1, "hidden_output": W2},
        "offsets": {"hidden": T1, "output": T2},
        "predictions": pred,
        "outputs": raw,
        "classes": labels,
        "accuracy": acc,
        "mse": mse,
        "iterations": it,
        "method": "two-layer perceptron trained by back-propagation, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 10.8, "
                  "eqs. (10.79)-(10.85)",
    })


rangayyan_ann_mlp = mlpbp  # pre-policy spelling


# -- rgbayes: Bayes minimum-error classifier.
def bayescls(likelihoods, priors=None):
    """Bayes decision functions, eq. (10.70).

        d_i(x) = p(x|C_i) P(C_i),  assign to the largest

    The MAXIMUM A POSTERIORI rule.  Dropping the prior and comparing the
    likelihoods alone is maximum likelihood, which is a different
    classifier and differs whenever the classes are unequally common --
    for a rare disease the prior is precisely what stops the classifier
    calling everything positive.

    The posterior probabilities are returned normalized, since the
    products themselves are not probabilities.
    """
    lk = aslist(likelihoods)
    m = len(lk)
    if m < 2:
        raise ValueError("need at least two classes")
    if any(v < 0 for v in lk):
        raise ValueError("a likelihood cannot be negative")
    if priors is None:
        pr = [1.0 / m] * m
    else:
        pr = aslist(priors)
        if len(pr) != m:
            raise ValueError("give one prior per class")
        if any(v < 0 for v in pr):
            raise ValueError("a prior cannot be negative")
        s = fsum(pr)
        if abs(s - 1.0) > 1e-9:
            raise ValueError("the priors must sum to 1, got %g" % s)
    d = [lk[i] * pr[i] for i in range(m)]
    tot = fsum(d)
    post = [v / tot for v in d] if tot > 0 else [0.0] * m
    mapc = max(range(m), key=lambda i: d[i])
    mlc = max(range(m), key=lambda i: lk[i])
    return RichResult(payload={
        "d": d, "posterior": post, "assigned": mapc,
        "maximum_likelihood_choice": mlc,
        "prior_changed_the_decision": mapc != mlc,
        "priors": pr, "uniform_priors": priors is None,
        "method": "Rangayyan (2024) eq. (10.70)"})


rangayyan_bayes_classifier = bayescls  # pre-policy spelling


# -- rgbayng: Bayes classifier for normal (Gaussian) patterns.
def bayesnorm(x, means, covs, priors=None, full=False):
    """Bayes classifier for normal patterns, eq. (10.72).

        d_i(x) = ln P(C_i) - (n/2) ln(2 pi) - (1/2) ln|C_i|
                 - (1/2) (x - m_i)^T C_i^-1 (x - m_i)

    The book takes logarithms at eq. (10.71) because the normal PDF is an
    exponential and ln is monotonic, so the ranking is unchanged while
    the arithmetic stops underflowing -- with a dozen features the raw
    densities are far below the smallest float.

    It then notes the (n/2) ln(2 pi) term "does not depend upon i" and
    drops it, giving eq. (10.73).  Dropping it is safe for CLASSIFYING
    and wrong for anything that reads the value as a log density; both
    forms are returned, ``full`` choosing which one is compared.

    The decision surfaces are hyperquadrics.  They reduce to hyperplanes
    exactly when all the covariance matrices are equal, which is the
    difference between this and ``qda``.
    """
    xs = aslist(x)
    ms = [aslist(v) for v in means]
    cs = [_mat(v) for v in covs]
    m = len(ms)
    if m < 2:
        raise ValueError("need at least two classes")
    if len(cs) != m:
        raise ValueError("give one covariance matrix per class")
    n = len(xs)
    if any(len(v) != n for v in ms):
        raise ValueError("every mean must match the length of x")
    if priors is None:
        pr = [1.0 / m] * m
    else:
        pr = aslist(priors)
        if len(pr) != m:
            raise ValueError("give one prior per class")
        if abs(fsum(pr) - 1.0) > 1e-9:
            raise ValueError("the priors must sum to 1")
    const = 0.5 * n * log(2.0 * pi)
    dfull, dshort = [], []
    for i in range(m):
        if pr[i] <= 0:
            dfull.append(float("-inf"))
            dshort.append(float("-inf"))
            continue
        det = _det(cs[i])
        if det <= 0:
            raise ValueError("covariance %d is not positive definite" % i)
        Ci = _inv(cs[i])
        d = [xs[j] - ms[i][j] for j in range(n)]
        quad = fsum(d[a] * fsum(Ci[a][b] * d[b] for b in range(n))
                    for a in range(n))
        short = log(pr[i]) - 0.5 * log(det) - 0.5 * quad
        dshort.append(short)
        dfull.append(short - const)
    use = dfull if full else dshort
    best = max(range(m), key=lambda i: use[i])
    equal = all(all(abs(cs[0][a][b] - cs[i][a][b]) < 1e-12
                    for a in range(n) for b in range(n))
                for i in range(m))
    return RichResult(payload={
        "d": use, "d_full": dfull, "d_dropped_constant": dshort,
        "assigned": best, "priors": pr,
        "constant_term": const,
        "surfaces_are_hyperquadrics": True,
        "linear_when_covariances_are_equal": equal,
        "log_form_avoids_underflow": True,
        "method": "Rangayyan (2024) eqs. (10.71)-(10.73)"})


rangayyan_bayes_gaussian = bayesnorm  # pre-policy spelling


# -- rgbbb: Bundle branch block (BBB) classification from ECG.
def bbb(qrsdur, criteria=None):
    """Apply the ECG decision rules for incomplete bundle-branch block.

    Why: bundle-branch block desynchronises ventricular contraction, and the
    ECG signature is a wider-than-normal QRS complex (100-120 ms or more) that
    may also be jagged or slurred.  The published diagnostic logic is a
    conjunction of duration and amplitude measurements on named leads, so a
    program can apply it once those measurements exist.  Rangayyan,
    *Biomedical Signal Analysis*, 3rd ed., Section 10.2.1.

    Incomplete **left** bundle-branch block requires all of:
    QRS duration >= 105 ms and <= 120 ms; QRS amplitude negative in V1 and V2;
    Q or S duration >= 80 ms in V1 and V2; no Q wave in any two of I, V5, V6;
    R duration > 60 ms in any two of I, aVL, V5, V6.

    Incomplete **right** bundle-branch block requires all of:
    QRS duration >= 91 ms and <= 120 ms; S duration >= 40 ms in any two of
    I, aVL, V4, V5, V6; and, in V1 or V2, either (R duration > 30 ms and
    R amplitude > 100 uV and no S wave) or the same three conditions on R'.

    Parameters
    ----------
    qrsdur : float
        Measured QRS duration in milliseconds.
    criteria : dict, optional
        The remaining measured conditions, as booleans.  Left-block keys:
        ``qrsneg_v1v2``, ``qsdur80_v1v2``, ``noq_two_of_i_v5_v6``,
        ``rdur60_two_of_i_avl_v5_v6``.  Right-block keys:
        ``sdur40_two_of_i_avl_v4_v5_v6``, ``r_v1v2``, ``rprime_v1v2``.
        Missing keys count as not satisfied.

    Returns
    -------
    RichResult
        Keys ``blocktype``, ``qrsdur``, ``wide``, ``left``, ``right``,
        ``satisfied``, ``method``.
    """
    try:
        qrsdur = float(qrsdur)
    except (TypeError, ValueError):
        raise ValueError("qrsdur must be a number of milliseconds")
    if not isfinite(qrsdur) or qrsdur <= 0.0:
        raise ValueError("qrsdur must be a positive, finite duration in ms")
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, dict):
        raise ValueError("criteria must be a dict of boolean measurements")

    def g(k):
        return bool(criteria.get(k, False))

    left_parts = {
        "qrs_105_to_120_ms": 105.0 <= qrsdur <= 120.0,
        "qrs_negative_in_v1_and_v2": g("qrsneg_v1v2"),
        "q_or_s_at_least_80_ms_in_v1_and_v2": g("qsdur80_v1v2"),
        "no_q_in_two_of_i_v5_v6": g("noq_two_of_i_v5_v6"),
        "r_over_60_ms_in_two_of_i_avl_v5_v6": g("rdur60_two_of_i_avl_v5_v6"),
    }
    right_parts = {
        "qrs_91_to_120_ms": 91.0 <= qrsdur <= 120.0,
        "s_at_least_40_ms_in_two_of_i_avl_v4_v5_v6":
            g("sdur40_two_of_i_avl_v4_v5_v6"),
        "r_or_rprime_pattern_in_v1_or_v2": g("r_v1v2") or g("rprime_v1v2"),
    }
    left = all(left_parts.values())
    right = all(right_parts.values())
    wide = qrsdur > 100.0

    if left and right:
        block = "criteria met for both left and right incomplete block"
    elif left:
        block = "incomplete left bundle-branch block"
    elif right:
        block = "incomplete right bundle-branch block"
    elif qrsdur > 120.0:
        block = "QRS wider than 120 ms, complete block not excluded"
    elif wide:
        block = "QRS wider than normal, block criteria not met"
    else:
        block = "no bundle-branch block by these criteria"

    return RichResult(payload={
        "blocktype": block,
        "qrsdur": qrsdur,
        "wide": wide,
        "left": left,
        "right": right,
        "satisfied": {"left": left_parts, "right": right_parts},
        "method": "incomplete bundle-branch block decision rules, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 10.2.1",
    })


rangayyan_bundle_branch_block = bbb  # pre-policy spelling


# -- rgbbnorm: Normal versus ectopic beat classification with LDA and Bayes.
def pvcbayes(features, labels, priors=None, query=None):
    """Bayes classification of ECG beats as normal or ectopic from [QRSTA, FF].

    Why: the linear rule of Section 10.11.1 commits to a hard boundary, while
    a Bayes classifier states the posterior odds and lets the prevalence of
    ectopy enter explicitly.  That matters here: in a real ECG the prior for a
    PVC is far below one half, and the book contrasts the equal-prior result
    with priors of 0.999 and 0.001.  Rangayyan, *Biomedical Signal Analysis*,
    3rd ed., Section 10.11.2, using the normal-pattern Bayes classifier of
    Section 10.6.2.

    Each feature is normalised by dividing by its standard deviation, as in the
    book, a 2-D Gaussian is fitted per class, and the beat is assigned to the
    class of largest posterior.  ``QRSTA`` is the area under the segmented,
    baseline-corrected and rectified QRS-T wave (Section 5.4.3) and ``FF`` is
    the form factor of eq. (5.26).

    Parameters
    ----------
    features : sequence of sequences
        One row per beat; columns are the features, canonically [QRSTA, FF].
    labels : sequence
        Class code per beat, 0 for normal and 1 for PVC.
    priors : sequence, optional
        Prior probabilities in class order.  Equal priors by default.
    query : sequence of sequences, optional
        Further feature rows to classify with the fitted model.

    Returns
    -------
    RichResult
        Keys ``predictions``, ``queryclass``, ``posterior``, ``means``,
        ``covariances``, ``scale``, ``confusion``, ``accuracy``,
        ``sensitivity``, ``specificity``, ``priors``, ``method``.
    """
    F = _bxmat(features, "features")
    y = [int(t) for t in _bxvec(labels, "labels")]
    if len(F) != len(y):
        raise ValueError("features and labels must have the same length")
    classes = sorted(set(y))
    if len(classes) != 2:
        raise ValueError("pvcbayes expects exactly two classes, e.g. 0 and 1")
    p = len(F[0])
    if priors is None:
        pri = [0.5, 0.5]
    else:
        pri = _bxvec(priors, "priors")
        if len(pri) != 2 or any(t < 0 for t in pri) or fsum(pri) <= 0:
            raise ValueError("priors must be two nonnegative numbers")
        s = fsum(pri)
        pri = [t / s for t in pri]

    scale = []
    for j in range(p):
        sd = _bxsd([F[i][j] for i in range(len(F))])
        scale.append(sd if sd > 0 else 1.0)
    Z = [[F[i][j] / scale[j] for j in range(p)] for i in range(len(F))]

    means, covs, invs, dets = [], [], [], []
    for c in classes:
        rows = [Z[i] for i in range(len(Z)) if y[i] == c]
        if len(rows) < p + 1:
            raise ValueError("each class needs more rows than features "
                             "to estimate a covariance matrix")
        mu, C = _bxcov(rows)
        for i in range(p):
            C[i][i] += 1e-9
        vals, vecs = _bxjacobi(C)
        if min(vals) <= 0.0:
            raise ValueError("a class covariance matrix is not positive definite")
        det = 1.0
        for t in vals:
            det *= t
        inv = [[fsum(vecs[i][k] * vecs[j][k] / vals[k] for k in range(p))
                for j in range(p)] for i in range(p)]
        means.append(mu)
        covs.append(C)
        invs.append(inv)
        dets.append(det)

    def post(z):
        dens = []
        for c in range(2):
            d = [z[j] - means[c][j] for j in range(p)]
            q = fsum(d[a] * invs[c][a][b] * d[b] for a in range(p) for b in range(p))
            dens.append(pri[c] * exp(-0.5 * q) / sqrt(((2.0 * pi) ** p) * dets[c]))
        tot = fsum(dens)
        if tot <= 0.0:
            return [0.5, 0.5]
        return [t / tot for t in dens]

    P = [post(z) for z in Z]
    pred = [classes[0] if t[0] >= t[1] else classes[1] for t in P]
    tp, tn, fp, fn = _bxconfusion([1 if t == classes[1] else 0 for t in y],
                                  [1 if t == classes[1] else 0 for t in pred])
    sen, spe, acc = _bxscores(tp, tn, fp, fn)

    qcls = None
    if query is not None:
        Q = _bxmat(query, "query")
        if len(Q[0]) != p:
            raise ValueError("query must have the same number of features")
        qcls = []
        for row in Q:
            z = [row[j] / scale[j] for j in range(p)]
            t = post(z)
            qcls.append(classes[0] if t[0] >= t[1] else classes[1])

    return RichResult(payload={
        "predictions": pred,
        "queryclass": qcls,
        "posterior": P,
        "means": means,
        "covariances": covs,
        "scale": scale,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": acc,
        "sensitivity": sen,
        "specificity": spe,
        "priors": pri,
        "classes": classes,
        "method": "Gaussian Bayes classifier on [QRSTA, FF] beat features, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 10.11.2 "
                  "with the normal-pattern classifier of Section 10.6.2",
    })


rangayyan_ecg_bbb_normal = pvcbayes  # pre-policy spelling


# -- rgbci: BCI EEG channel selection via NMF spatial decomposition.
def bcichsel(trials, nselect, rank=4, maxiter=200, tol=1e-8, seed=1):
    """Select and weight EEG channels for a motor-imagery BCI, via NMF.

    Why: a practical brain-computer interface has to run under hardware
    complexity constraints, and the optimal channel set is subject specific,
    so a data-driven selector beats a fixed montage.  This is the whole
    application of Rangayyan, *Biomedical Signal Analysis*, 3rd ed.,
    Section 9.12 (channel-selection core in Section 9.12.1).

    The per-trial channel covariance matrix of eq. (9.94) is factorised by
    NMF, each row of the basis matrix W is min-max normalised by eq. (9.95),
    and its RMS deviation from the reference vector of all 0.5 is taken by
    eq. (9.96) as the channel score.  The final step of Section 9.12.1 is
    applied here: the estimated channel weights multiply the corresponding
    selected EEG channels, and the weighted channels are returned.

    Parameters
    ----------
    trials : sequence of sequences
        One trial, as an N x T matrix: N EEG channels by T samples.
    nselect : int
        How many channels to keep.
    rank : int
        NMF factorisation rank r; must be at least 3, since r = 2 makes the
        score of eq. (9.96) identically 0.5 for every channel.
    maxiter, tol : int, float
        Multiplicative-update budget and convergence tolerance.
    seed : int
        Seed for the deterministic NMF initialisation.

    Returns
    -------
    RichResult
        Keys ``selected``, ``weights``, ``rmsd``, ``weighted``, ``W``, ``H``,
        ``covariance``, ``error``, ``method``.
    """
    X = _bxmat(trials, "trials")
    nch = len(X)
    if nch < 2:
        raise ValueError("need at least two EEG channels")
    nselect = int(nselect)
    if not (1 <= nselect <= nch):
        raise ValueError("nselect must satisfy 1 <= nselect <= number of channels")
    rank = int(rank)
    if rank < 3:
        raise ValueError("rank must be at least 3: with r = 2 the min-max "
                         "normalisation of eq. (9.95) maps every basis row to "
                         "{0, 1}, so the RMS deviation of eq. (9.96) is exactly "
                         "0.5 for every channel and ranks nothing")

    mu, C = _bxcov(_bxtr(X))
    shift = min(min(r) for r in C)
    V = [[t - shift for t in r] for r in C] if shift < 0.0 else [list(r) for r in C]
    W, H, err, _ = _bxnmfmu(V, rank, maxiter, tol, seed, "ls")

    rmsd, Wn = [], []
    for j in range(nch):
        row = W[j]
        lo, hi = min(row), max(row)
        nr = [0.5] * len(row) if hi - lo <= 0 else [(t - lo) / (hi - lo) for t in row]
        Wn.append(nr)
        rmsd.append(sqrt(fsum((t - 0.5) ** 2 for t in nr) / len(nr)))

    order = sorted(range(nch), key=lambda i: (-rmsd[i], i))
    sel = sorted(order[:nselect])
    weighted = [[rmsd[i] * t for t in X[i]] for i in sel]

    return RichResult(payload={
        "selected": sel,
        "weights": [rmsd[i] for i in sel],
        "rmsd": rmsd,
        "normalized": Wn,
        "weighted": weighted,
        "W": W,
        "H": H,
        "covariance": C,
        "error": err,
        "method": "NMF-based EEG channel selection and weighting for BCI, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 9.12.1, "
                  "eqs. (9.94)-(9.96)",
    })


rangayyan_bci_nmf = bcichsel  # pre-policy spelling


# -- rgbhatt: Bhattacharyya distance for class separability.
def normdist(m1, m2, s1, s2):
    """Normalized distance between two class PDFs, eq. (10.112).

        d_n = |m1 - m2| / (sigma1 + sigma2)

    Note the denominator is the SUM of the standard deviations, not the
    root of the sum of their squares -- this is not the Fisher criterion.
    The book states its limitation outright: d_n = 0 whenever m1 = m2,
    however different the dispersions are, so two classes distinguished
    only by their variance score zero separability on this measure.  The
    divergence of eq. (10.115) is what does not have that blind spot.
    """
    a, b = float(m1), float(m2)
    p, q = float(s1), float(s2)
    if p < 0 or q < 0:
        raise ValueError("a standard deviation cannot be negative")
    den = p + q
    if den <= 0:
        raise ValueError("both standard deviations are zero; the "
                         "normalized distance is undefined")
    return RichResult(payload={
        "dn": abs(a - b) / den, "mean_difference": abs(a - b),
        "sd_sum": den,
        "blind_to_variance_when_means_match": abs(a - b) < 1e-300,
        "denominator_is_the_sum_not_the_quadrature_sum": True,
        "method": "Rangayyan (2024) eq. (10.112)"})


def divergence(m1, m2, C1, C2):
    """Divergence between two multivariate Gaussian PDFs, eq. (10.117).

        D_ij = (1/2) tr[ (C_i - C_j)(C_j^-1 - C_i^-1) ]
             + (1/2) tr[ (C_i^-1 + C_j^-1)(m_i - m_j)(m_i - m_j)^T ]

    the closed form of the symmetric divergence defined at eq. (10.115),
    D_ij = E[l'_ij | C_i] + E[l'_ji | C_j].

    The second term is the one resembling the normalized distance of
    eq. (10.112) and vanishes for equal means; the FIRST term does not,
    so unlike d_n the divergence still separates classes that differ only
    in their covariance.  The book's stated properties are checked and
    returned: D > 0, D_ii = 0, and D_ij = D_ji.

    This is the measure Rangayyan uses.  Bhattacharyya distance does not
    appear in this book.
    """
    a, b = aslist(m1), aslist(m2)
    A, B = _mat(C1), _mat(C2)
    p = len(a)
    if len(b) != p:
        raise ValueError("the two mean vectors must have the same length")
    if len(A) != p or len(B) != p or any(len(r) != p for r in A + B):
        raise ValueError("the covariance matrices must be %d x %d" % (p, p))
    Ai, Bi = _inv(A), _inv(B)
    diff = [[A[i][j] - B[i][j] for j in range(p)] for i in range(p)]
    invd = [[Bi[i][j] - Ai[i][j] for j in range(p)] for i in range(p)]
    term1 = 0.5 * _trace(_matmul(diff, invd))
    dm = [a[i] - b[i] for i in range(p)]
    outer = [[dm[i] * dm[j] for j in range(p)] for i in range(p)]
    summ = [[Ai[i][j] + Bi[i][j] for j in range(p)] for i in range(p)]
    term2 = 0.5 * _trace(_matmul(summ, outer))
    D = term1 + term2
    return RichResult(payload={
        "divergence": D, "covariance_term": term1, "mean_term": term2,
        "nonnegative": D >= -1e-9,
        "symmetric": True, "zero_for_identical_pdfs": abs(D) < 1e-9,
        "separates_equal_means_via_the_covariance_term": abs(term1) > 1e-12,
        "additive_over_independent_features": True,
        "method": "Rangayyan (2024) eqs. (10.115)-(10.117)"})


def divav(means, covs):
    """Average pairwise divergence over m classes, following eq. (10.117).

    The book averages the pairwise divergences to obtain a single measure
    across all classes.  Averaging hides a badly separated PAIR behind
    well separated ones, so the minimum pairwise value is returned too --
    it is the pair that will actually be confused.
    """
    ms = [aslist(v) for v in means]
    cs = [_mat(v) for v in covs]
    m = len(ms)
    if m < 2:
        raise ValueError("need at least two classes")
    if len(cs) != m:
        raise ValueError("give one covariance matrix per class")
    vals, pairs = [], []
    for i in range(m):
        for j in range(i + 1, m):
            d = divergence(ms[i], ms[j], cs[i], cs[j])["divergence"]
            vals.append(d)
            pairs.append((i, j, d))
    worst = min(pairs, key=lambda t: t[2])
    return RichResult(payload={
        "average": fsum(vals) / len(vals), "pairwise": pairs,
        "minimum": worst[2], "worst_pair": (worst[0], worst[1]),
        "n_classes": m, "n_pairs": len(vals),
        "average_hides_the_worst_pair": True,
        "method": "Rangayyan (2024) Section 10.10.1 (average divergence)"})


def kld(p1, p2):
    """Kullback-Leibler distance or divergence, eq. (5.33).

        KLD(p1, p2) = sum_l p2(x_l) ln[ p2(x_l) / p1(x_l) ]

    NOTE THE ARGUMENT ORDER.  The book weights by the SECOND pdf, which
    makes its KLD(p1, p2) equal to D_KL(p2 || p1) in the standard
    notation -- the REVERSE of what most texts and libraries mean by
    KL(p, q).  That is the single most likely way to get a wrong number
    out of this family, so both directions are returned and the
    convention is stated in the payload.

    The arithmetic is ``morie.fn.kldiv.kl_divergence`` called with the
    arguments swapped; this is NOT a second implementation.

    KLD is not symmetric.  The symmetric combination is the divergence of
    eq. (10.115), which is exactly KLD(p1, p2) + KLD(p2, p1).

    The book uses this as a FEATURE: Rangayyan and Wu computed the KLD
    between the PDF of a signal to be classified and Parzen-window PDF
    models of the normal and abnormal VAG classes, reaching 73 per cent
    classification with the KLD alone.
    """
    from .kldiv import kl_divergence
    a, b = aslist(p1), aslist(p2)
    if len(a) != len(b):
        raise ValueError("the two PDFs must be sampled on the same grid")
    if not a:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in a) or any(v < 0 for v in b):
        raise ValueError("a PDF cannot be negative")
    bad = [i for i in range(len(a)) if b[i] > 0 and a[i] <= 0]
    if bad:
        raise ValueError("p1 vanishes at %d bin(s) where p2 does not; "
                         "the KLD is unbounded there" % len(bad))
    fwd = float(kl_divergence(b, a).estimate)      # note the swap
    rev = float(kl_divergence(a, b).estimate)
    return RichResult(payload={
        "kld": fwd, "reversed": rev, "symmetric_sum": fwd + rev,
        "asymmetric": abs(fwd - rev) > 1e-12,
        "weighted_by_the_second_pdf": True,
        "book_order_is_the_reverse_of_the_standard": True,
        "standard_notation": "KLD(p1, p2) here is D_KL(p2 || p1)",
        "symmetric_sum_is_the_divergence_of_eq_10_115": True,
        "nonnegative": fwd >= -1e-12,
        "delegates_to": "morie.fn.kldiv.kl_divergence",
        "method": "Rangayyan (2024) eq. (5.33)"})


def pdfoverlap(p1, p2):
    """Bhattacharyya coefficient, the OVERLAP between two PDFs.

        BC(p1, p2) = sum_l sqrt( p1(x_l) p2(x_l) )

    Bounded in [0, 1]: 1 when the two PDFs are identical, 0 when their
    supports do not touch.  This is the quantity the Bhattacharyya
    DISTANCE is built from, D_B = -ln BC, and it is what makes the error
    bound work -- the overlap of the two class-conditional densities IS
    the region where the optimal classifier must make mistakes.

    NOT FROM THIS BOOK; see ``gaussoverlap``.
    """
    a, b = aslist(p1), aslist(p2)
    if len(a) != len(b):
        raise ValueError("the two PDFs must be sampled on the same grid")
    if not a:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in a) or any(v < 0 for v in b):
        raise ValueError("a PDF cannot be negative")
    bc = fsum(sqrt(a[i] * b[i]) for i in range(len(a)))
    return RichResult(payload={
        "coefficient": bc, "overlap": bc,
        "distance": (-log(bc)) if bc > 0 else float("inf"),
        "identical": abs(bc - 1.0) < 1e-12,
        "disjoint": bc <= 1e-15,
        "in_unit_interval": -1e-12 <= bc <= 1.0 + 1e-12,
        "the_overlap_is_where_errors_must_happen": True,
        "not_from_this_book": True,
        "reference": "Bhattacharyya A. On a measure of divergence "
                     "between two statistical populations defined by "
                     "their probability distributions. Bulletin of the "
                     "Calcutta Mathematical Society 35:99-109, 1943 "
                     "(Zbl 0063.00364).",
        "method": "Bhattacharyya coefficient; Rangayyan (2024) uses the "
                  "KLD of eq. (5.33) and the divergence of eq. (10.115)"})


def chernoff(p1, p2, alpha=None, n_grid=201):
    """Chernoff coefficient and information.

        rho_a(p1, p2) = sum_l p1(x_l)^a p2(x_l)^(1-a),   0 <= a <= 1
        C(p1, p2)     = -ln min_a rho_a

    The Bhattacharyya coefficient is exactly this at a = 1/2, which is
    the relationship the whole family turns on: BC = rho_{1/2}.  Leaving
    ``alpha`` unset searches the interval for the minimizing a, which is
    what makes the Chernoff bound TIGHTER than the Bhattacharyya bound --
    the latter is the same bound evaluated at a fixed a = 1/2 instead of
    the best one.

    The bound it gives is P_e <= P1^a P2^(1-a) rho_a for ANY a, so every
    a yields a valid bound and the minimum yields the best of them.  At
    equal priors and a = 1/2 this collapses to Kailath's
    sqrt(P1 P2) exp(-D_B).

    NOT FROM RANGAYYAN (2024); the book uses the KLD of eq. (5.33).
    """
    a, b = aslist(p1), aslist(p2)
    if len(a) != len(b):
        raise ValueError("the two PDFs must be sampled on the same grid")
    if not a:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in a) or any(v < 0 for v in b):
        raise ValueError("a PDF cannot be negative")

    def rho(t):
        return fsum((a[i] ** t) * (b[i] ** (1.0 - t)) for i in range(len(a))
                    if a[i] > 0 and b[i] > 0)

    if alpha is not None:
        av = float(alpha)
        if not 0.0 <= av <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        best_a, best_rho = av, rho(av)
        searched = False
    else:
        m = int(n_grid)
        if m < 3:
            raise ValueError("need at least three grid points")
        grid = [i / (m - 1) for i in range(m)]
        vals = [rho(t) for t in grid]
        k = min(range(m), key=lambda i: vals[i])
        best_a, best_rho = grid[k], vals[k]
        searched = True
    bc = rho(0.5)
    return RichResult(payload={
        "coefficient": best_rho, "alpha": best_a,
        "information": (-log(best_rho)) if best_rho > 0 else float("inf"),
        "bhattacharyya_coefficient": bc,
        "bhattacharyya_is_alpha_one_half": True,
        "alpha_searched": searched,
        "at_least_as_tight_as_bhattacharyya": best_rho <= bc + 1e-12,
        "reference": "Chernoff H. A measure of asymptotic efficiency for "
                     "tests of a hypothesis based on the sum of "
                     "observations. Annals of Mathematical Statistics "
                     "23(4):493-507, 1952, doi:10.1214/aoms/1177729330. "
                     "The alpha = 1/2 identity is Nielsen and Nock, "
                     "Pattern Recognition Letters, 2014.",
        "not_from_this_book": True,
        "method": "Chernoff alpha-coefficient and information"})


def hellinger(p1, p2):
    """Hellinger distance, in the class-separability framing.

        H(p1, p2) = sqrt( 1 - BC(p1, p2) )

    The arithmetic is ``morie.fn.helld.hellinger_dist``.  This is NOT a
    second implementation -- it is that one with the Bhattacharyya
    relationship attached, because in a classification context the point
    of H is its connection to the overlap BC.

    Unlike the Bhattacharyya distance -ln BC, this is a TRUE METRIC:
    bounded in [0, 1], symmetric, and it satisfies the triangle
    inequality, which -ln BC does not.  That is the reason to reach for
    it -- anything needing a metric over distributions needs this one.

    NOT FROM RANGAYYAN (2024).
    """
    from .helld import hellinger_dist
    a, b = aslist(p1), aslist(p2)
    if len(a) != len(b):
        raise ValueError("the two PDFs must be sampled on the same grid")
    if not a:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in a) or any(v < 0 for v in b):
        raise ValueError("a PDF cannot be negative")
    h = float(hellinger_dist(a, b).estimate)
    sa, sb = fsum(a), fsum(b)
    if sa <= 0 or sb <= 0:
        raise ValueError("a PDF has no mass")
    bc = fsum(sqrt((a[i] / sa) * (b[i] / sb)) for i in range(len(a)))
    return RichResult(payload={
        "hellinger": h, "squared": h * h,
        "bhattacharyya_coefficient": bc,
        "identity_h2_equals_one_minus_bc": abs(h * h - (1.0 - bc)) < 1e-12,
        "is_a_true_metric": True,
        "satisfies_the_triangle_inequality": True,
        "bhattacharyya_distance_does_not": True,
        "normalization": "one half; unnormalized gives 2(1 - BC)",
        "in_unit_interval": -1e-12 <= h <= 1.0 + 1e-12,
        "delegates_to": "morie.fn.helld.hellinger_dist",
        "reference": "Hellinger E. Neue Begruendung der Theorie "
                     "quadratischer Formen von unendlichvielen "
                     "Veraenderlichen. Journal fuer die reine und "
                     "angewandte Mathematik 136:210-271, 1909, "
                     "doi:10.1515/crll.1909.136.210.",
        "not_from_this_book": True,
        "method": "Hellinger distance, H^2 = 1 - BC"})


def gaussoverlap(m1, m2, C1, C2):
    """Bhattacharyya distance between two multivariate Gaussians.

        D_B = (1/8) (m1-m2)^T [(C1+C2)/2]^-1 (m1-m2)
              + (1/2) ln( |(C1+C2)/2| / sqrt(|C1| |C2|) )

    NOT FROM THIS BOOK.  A full-text search of the 2024 third edition --
    Rangayyan and Krishnan -- finds no occurrence of "Bhattacharyya", nor
    of Chernoff or Hellinger.  What the book gives instead is the KLD of
    eq. (5.33) and the divergence of eqs. (10.115)-(10.117), which is the
    symmetric sum of the two KLDs.

    WHAT IT IS FOR, and why it is kept: D_B = -ln BC where BC is the
    Bhattacharyya coefficient, the overlap between the two
    class-conditional densities.  That overlap is precisely the region in
    which the optimal classifier is forced to err, which is why D_B
    BOUNDS the Bayes error, P_e <= sqrt(P1 P2) exp(-D_B), and the
    divergence does not bound anything.  So the two measures answer
    different questions: the divergence says how far apart the classes
    are, the Bhattacharyya distance says how well any classifier could
    possibly do.  Use ``divergence`` when following the book, this when
    an error bound is wanted.  It is implemented here because the name was already exposed
    and because it is a correct and standard measure -- and because it,
    unlike the divergence, bounds the Bayes error directly.  Prefer
    ``divergence`` when following the book.
    """
    a, b = aslist(m1), aslist(m2)
    A, B = _mat(C1), _mat(C2)
    p = len(a)
    if len(b) != p:
        raise ValueError("the two mean vectors must have the same length")
    if len(A) != p or len(B) != p or any(len(r) != p for r in A + B):
        raise ValueError("the covariance matrices must be %d x %d" % (p, p))
    M = [[0.5 * (A[i][j] + B[i][j]) for j in range(p)] for i in range(p)]
    Mi = _inv(M)
    dm = [a[i] - b[i] for i in range(p)]
    quad = fsum(dm[i] * fsum(Mi[i][j] * dm[j] for j in range(p))
                for i in range(p))
    dA, dB, dM = _det(A), _det(B), _det(M)
    if dA <= 0 or dB <= 0 or dM <= 0:
        raise ValueError("a covariance matrix is not positive definite")
    return RichResult(payload={
        "bhattacharyya": 0.125 * quad + 0.5 * log(dM / sqrt(dA * dB)),
        "mean_term": 0.125 * quad,
        "covariance_term": 0.5 * log(dM / sqrt(dA * dB)),
        "not_from_this_book": True,
        "book_uses_divergence_eq_10_115": True,
        "reference": "Bhattacharyya A. Bulletin of the Calcutta "
                     "Mathematical Society 35:99-109, 1943; the "
                     "Gaussian closed form and the error bound are "
                     "Kailath T, The divergence and Bhattacharyya "
                     "distance measures in signal selection, IEEE "
                     "Transactions on Communication Technology "
                     "15(1):52-60, 1967, doi:10.1109/TCOM.1967.1089532.",
        "method": "standard Bhattacharyya distance for Gaussians; "
                  "Rangayyan (2024) uses eqs. (10.112) and (10.115) "
                  "instead"})


rangayyan_bhattacharyya = divergence  # pre-policy spelling


# -- rgbp: Basis pursuit: L1 minimization for sparse representation.
def bpursuit(x, D, lam=0.01, maxiter=2000, tol=1e-10):
    """Basis-pursuit denoising: L1-penalised expansion in an overcomplete dictionary.

    Why: greedy pursuit fixes each atom the moment it is chosen, so an early
    mistake is never revisited.  Basis pursuit instead solves a convex problem
    whose L1 penalty produces a sparse expansion while every coefficient stays
    free to the end, which is the right tool when dictionary atoms are strongly
    correlated.

    Minimises 0.5 * ||x - D' a||^2 + lam * ||a||_1 by iterative soft
    thresholding: a <- soft(a + D (x - D' a) / L, lam / L) with L the squared
    spectral norm bound, estimated here by power iteration on D D'.

    Not from Rangayyan: *Biomedical Signal Analysis*, 3rd ed. covers matching
    pursuit (Section 9.3) and EMD-based dictionary learning (Section 9.5) but
    does not present basis pursuit.  Primary sources are Chen, Donoho and
    Saunders, "Atomic decomposition by basis pursuit", SIAM Journal on
    Scientific Computing 20(1):33-61, 1998, for the L1 formulation, and
    Daubechies, Defrise and De Mol, Communications on Pure and Applied
    Mathematics 57(11):1413-1457, 2004, for the thresholded-Landweber solver.

    Parameters
    ----------
    x : sequence
        Signal to represent.
    D : sequence of sequences
        Dictionary, one atom per row, each the same length as x.
    lam : float
        L1 penalty weight; larger values give sparser expansions.
    maxiter : int
        Maximum thresholding iterations.
    tol : float
        Stop when the largest coefficient change falls below this.

    Returns
    -------
    RichResult
        Keys ``alpha``, ``support``, ``reconstruction``, ``residual``,
        ``l1norm``, ``objective``, ``iterations``, ``method``.
    """
    x = _bxvec(x, "x")
    A = _bxmat(D, "D")
    n = len(x)
    if any(len(a) != n for a in A):
        raise ValueError("every dictionary atom must have the same length as x")
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lam must be nonnegative")
    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    m = len(A)
    v = [1.0 / sqrt(m)] * m
    L = 1.0
    for _ in range(60):
        w = _bxmv(A, _bxmv(_bxtr(A), v))
        nw = _bxnrm(w)
        if nw <= 1e-300:
            break
        v = [t / nw for t in w]
        L = nw
    L = max(L, 1e-12)

    a = [0.0] * m
    it = 0
    for it in range(1, maxiter + 1):
        approx = _bxmv(_bxtr(A), a)
        r = [x[i] - approx[i] for i in range(n)]
        g = _bxmv(A, r)
        shift = 0.0
        for j in range(m):
            z = a[j] + g[j] / L
            th = lam / L
            new = 0.0 if abs(z) <= th else (z - th if z > 0 else z + th)
            shift = max(shift, abs(new - a[j]))
            a[j] = new
        if shift <= tol:
            break

    approx = _bxmv(_bxtr(A), a)
    r = [x[i] - approx[i] for i in range(n)]
    l1 = fsum(abs(t) for t in a)
    obj = 0.5 * fsum(t * t for t in r) + lam * l1
    return RichResult(payload={
        "alpha": a,
        "support": [j for j in range(m) if a[j] != 0.0],
        "reconstruction": approx,
        "residual": r,
        "l1norm": l1,
        "objective": obj,
        "iterations": it,
        "method": "basis-pursuit denoising by iterative soft thresholding; "
                  "Chen, Donoho and Saunders, SIAM J. Sci. Comput. 20(1):33-61, "
                  "1998; solver of Daubechies, Defrise and De Mol, Comm. Pure "
                  "Appl. Math. 57(11):1413-1457, 2004 (not covered by Rangayyan)",
    })


rangayyan_basis_pursuit = bpursuit  # pre-policy spelling


# -- rgcad: Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate.
def cadpipe(features, labels, k=5, standardize=True):
    """Run and validate a screening CAD pipeline end to end.

    Why: a computer-aided diagnosis system is not a classifier but a chain,
    and the accuracy quoted for that chain is only meaningful if the test
    patterns were never seen during training.  This runs the whole chain --
    feature standardisation, a linear discriminant designed on the training
    partition only, and k-fold validation -- and reports the diagnostic
    measures the clinical reader needs.  Rangayyan, *Biomedical Signal
    Analysis*, 3rd ed., Chapter 10; training and test steps, Section 10.10.3;
    diagnostic measures, Section 10.9.

    The classifier is the prototype (mean) discriminant of Section 10.4.1: a
    pattern is assigned to the class whose training-set mean vector is nearer,
    which is the decision boundary given by the normal bisector of the segment
    joining the two prototypes.  Sensitivity, specificity and accuracy follow
    eqs. (10.100), (10.101) and (10.103); prevalence-weighted accuracy follows
    eq. (10.102).

    Parameters
    ----------
    features : sequence of sequences
        One pattern vector per subject or signal.
    labels : sequence
        Binary class code, 0 for without the disease and 1 for with it.
    k : int
        Number of cross-validation folds; ``k`` equal to the sample size gives
        leave-one-out validation.
    standardize : bool
        Divide each feature by its training-partition standard deviation.

    Returns
    -------
    RichResult
        Keys ``accuracy``, ``sensitivity``, ``specificity``,
        ``weightedaccuracy``, ``confusion``, ``predictions``, ``folds``,
        ``prevalence``, ``method``.
    """
    F = _bxmat(features, "features")
    y = [int(t) for t in _bxvec(labels, "labels")]
    if len(F) != len(y):
        raise ValueError("features and labels must have the same length")
    if set(y) != {0, 1}:
        raise ValueError("labels must contain both 0 (without) and 1 (with) the disease")
    n, p = len(F), len(F[0])
    k = int(k)
    if not (2 <= k <= n):
        raise ValueError("k must satisfy 2 <= k <= number of patterns")

    folds = [[] for _ in range(k)]
    for c in (0, 1):
        members = [i for i in range(n) if y[i] == c]
        for j, i in enumerate(members):
            folds[j % k].append(i)
    if any(not f for f in folds):
        raise ValueError("k is too large: some fold is empty")

    pred = [None] * n
    for f in range(k):
        test = set(folds[f])
        train = [i for i in range(n) if i not in test]
        if len({y[i] for i in train}) < 2:
            raise ValueError("a training partition lost a class; reduce k")
        sc = [1.0] * p
        if standardize:
            for j in range(p):
                sd = _bxsd([F[i][j] for i in train])
                sc[j] = sd if sd > 0 else 1.0
        proto = {}
        for c in (0, 1):
            rows = [i for i in train if y[i] == c]
            proto[c] = [fsum(F[i][j] / sc[j] for i in rows) / len(rows)
                        for j in range(p)]
        for i in test:
            z = [F[i][j] / sc[j] for j in range(p)]
            d0 = fsum((z[j] - proto[0][j]) ** 2 for j in range(p))
            d1 = fsum((z[j] - proto[1][j]) ** 2 for j in range(p))
            pred[i] = 1 if d1 < d0 else 0

    tp, tn, fp, fn = _bxconfusion(y, pred)
    sen, spe, acc = _bxscores(tp, tn, fp, fn)
    prev = fsum(1.0 for t in y if t == 1) / n
    wacc = sen * prev + spe * (1.0 - prev)
    return RichResult(payload={
        "accuracy": acc,
        "sensitivity": sen,
        "specificity": spe,
        "weightedaccuracy": wacc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "predictions": pred,
        "folds": [sorted(f) for f in folds],
        "prevalence": prev,
        "method": "cross-validated prototype-discriminant CAD pipeline, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Sections 10.4.1 "
                  "and 10.10.3, scored by eqs. (10.100)-(10.103)",
    })


rangayyan_cad_pipeline = cadpipe  # pre-policy spelling


# -- rgcnn: 1D CNN for biomedical signal classification.
def cnnsig(x, kernels, bias=None, pool=2, dense=None):
    """Forward pass of a 1-D convolutional feature extractor for a biomedical signal.

    Why: in a deep network each hidden node acts as a learnable kernel or
    adaptive filter, and stacking them lets the network learn several levels
    of feature representation from the signal itself instead of from a
    hand-designed feature list.  Rangayyan, *Biomedical Signal Analysis*, 3rd
    ed., Section 10.8.2 describes this in prose and names CNNs as the common
    deep-learning model, but gives no equations for the convolution, the
    rectifier or the pooling stage; the layer definitions used here are those
    of LeCun, Bengio and Hinton, "Deep learning", Nature 521(7553):436-444,
    2015, which is reference [35] of that section.

    Each kernel is cross-correlated with the signal, rectified by
    max(0, .), and max-pooled by a factor ``pool``.  The concatenated pooled
    maps form the feature vector; if ``dense`` is given, that vector is passed
    through a linear output layer and a softmax to give class scores.

    Parameters
    ----------
    x : sequence
        Input signal.
    kernels : sequence of sequences
        One filter per row.
    bias : sequence, optional
        One bias per kernel; zeros by default.
    pool : int
        Max-pooling factor.
    dense : sequence of sequences, optional
        Output-layer weights, one row per class, each of length equal to the
        pooled feature vector.

    Returns
    -------
    RichResult
        Keys ``maps``, ``pooled``, ``features``, ``scores``, ``predicted``,
        ``method``.
    """
    x = _bxvec(x, "x")
    K = _bxmat(kernels, "kernels")
    pool = int(pool)
    if pool < 1:
        raise ValueError("pool must be a positive integer")
    if any(len(a) > len(x) for a in K):
        raise ValueError("every kernel must be no longer than the signal")
    if bias is None:
        b = [0.0] * len(K)
    else:
        b = _bxvec(bias, "bias")
        if len(b) != len(K):
            raise ValueError("bias must have one entry per kernel")

    maps, pooled = [], []
    for ki in range(len(K)):
        w = K[ki]
        m = len(w)
        conv = [max(0.0, fsum(w[j] * x[i + j] for j in range(m)) + b[ki])
                for i in range(len(x) - m + 1)]
        maps.append(conv)
        pl = [max(conv[i:i + pool]) for i in range(0, len(conv) - pool + 1, pool)]
        if not pl:
            pl = [max(conv)]
        pooled.append(pl)

    feat = [t for pl in pooled for t in pl]
    scores, best = None, None
    if dense is not None:
        Wd = _bxmat(dense, "dense")
        if len(Wd[0]) != len(feat):
            raise ValueError("dense rows must match the pooled feature length "
                             "of %d" % len(feat))
        z = _bxmv(Wd, feat)
        mx = max(z)
        e = [exp(t - mx) for t in z]
        s = fsum(e)
        scores = [t / s for t in e]
        best = max(range(len(scores)), key=lambda i: scores[i])

    return RichResult(payload={
        "maps": maps,
        "pooled": pooled,
        "features": feat,
        "scores": scores,
        "predicted": best,
        "method": "1-D convolution, rectifier and max-pooling forward pass; "
                  "architecture per LeCun, Bengio and Hinton, Nature "
                  "521(7553):436-444, 2015, cited as ref. [35] of Rangayyan "
                  "Biomedical Signal Analysis 3rd ed. Section 10.8.2, which "
                  "gives no equations for these layers",
    })


rangayyan_cnn_signal = cnnsig  # pre-policy spelling


# -- rgecgfe: Single-channel fetal ECG extraction using NMF/ICA.
def fecgnmf(x, fs, nwin=64, hop=None, rank=4, lam=0.0, maxiter=150,
            taum=0.6, tauf=0.45, seed=1):
    """Separate the fetal ECG from a single-channel abdominal ECG by NMF.

    Why: the fetal and maternal ECG overlap in the spectrum, so no linear
    filter separates them, and the usual blind-source-separation route needs
    several abdominal leads.  A single lead is what a mother can wear at home,
    so the separation has to come from structure in a time-frequency matrix
    rather than from spatial diversity.  Rangayyan, *Biomedical Signal
    Analysis*, 3rd ed., Section 9.11.

    The mixture model is eq. (9.88), x_a(t) = x_m(t) + x_f(t) + n(t).  The
    signal is converted to an M x N time-frequency matrix V by STFT and
    factorised.  Rows of the activation matrix H are normalised, thresholded
    at T = tau * A_max by eq. (9.91), and the peaks above threshold are
    counted; with the counts sorted in ascending order the smallest belongs to
    the maternal activation and the next to the fetal activation, because the
    fetal heart rate is the higher of the two.  The chosen components are
    turned back into signals through a soft mask and the inverse STFT.

    Sparse NMF updates follow eqs. (9.89) and (9.90) when ``lam`` is positive;
    with ``lam`` zero the standard multiplicative updates (9.49) and (9.50)
    are used.  Rangayyan reports r = 4 as empirically best for this task.

    Parameters
    ----------
    x : sequence
        Single-channel abdominal ECG, already baseline- and mains-filtered.
    fs : float
        Sampling rate in Hz.
    nwin, hop : int
        STFT window length and hop; hop defaults to ``nwin // 2``.
    rank : int
        Factorisation rank r.
    lam : float
        Sparsity parameter of eq. (9.89); zero selects the standard updates.
    maxiter : int
        Update iterations.
    taum, tauf : float
        Threshold fractions of eq. (9.91); the book uses 0.6 for the maternal
        and 0.45 for the fetal activation.
    seed : int
        Seed for the deterministic initialisation.

    Returns
    -------
    RichResult
        Keys ``fetal``, ``maternal``, ``peaks``, ``fetalrow``,
        ``maternalrow``, ``W``, ``H``, ``error``, ``method``.
    """
    x = _bxvec(x, "x")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be a positive sampling rate in Hz")
    nwin = int(nwin)
    hop = nwin // 2 if hop is None else int(hop)
    if not (0.0 < taum <= 1.0) or not (0.0 < tauf <= 1.0):
        raise ValueError("taum and tauf must lie in (0, 1]")
    if float(lam) < 0.0:
        raise ValueError("lam must be nonnegative")

    re_f, im_f, mag_f, win = _bxstft(x, nwin, hop)
    V = _bxtr(mag_f)
    W, H, err, _ = _bxnmfmu(V, rank, maxiter, 1e-10, seed,
                            "ls" if float(lam) == 0.0 else "kld")
    if float(lam) > 0.0:
        eps = 1e-12
        for _ in range(20):
            R = _bxmm(W, H)
            for a in range(len(H)):
                for b in range(len(H[0])):
                    num = fsum(V[i][b] * W[i][a] for i in range(len(V)))
                    den = fsum(R[i][b] * W[i][a] for i in range(len(V))) + float(lam)
                    H[a][b] *= num / (den + eps)

    def peakcount(row, tau):
        m = max(abs(t) for t in row)
        if m <= 0.0:
            return 0
        nr = [abs(t) / m for t in row]
        thr = tau * 1.0
        return sum(1 for i in range(1, len(nr) - 1)
                   if nr[i] > thr and nr[i] >= nr[i - 1] and nr[i] > nr[i + 1])

    counts = [(peakcount(H[a], tauf), a) for a in range(len(H))]
    counts.sort()
    mrow = counts[0][1]
    frow = counts[1][1] if len(counts) > 1 else counts[0][1]
    mcount = peakcount(H[mrow], taum)
    peaks = {"per_row": {a: c for c, a in counts},
             "maternal_row_count_at_taum": mcount}

    def rebuild(row):
        R = _bxmm(W, H)
        C = [[W[i][row] * H[row][j] for j in range(len(H[0]))]
             for i in range(len(W))]
        ren, imn = [], []
        for fi in range(len(mag_f)):
            rr, ii = [], []
            for k in range(len(mag_f[0])):
                tot = R[k][fi]
                msk = C[k][fi] / tot if tot > 1e-12 else 0.0
                rr.append(re_f[fi][k] * msk)
                ii.append(im_f[fi][k] * msk)
            ren.append(rr)
            imn.append(ii)
        return _bxistft(ren, imn, nwin, hop, win, len(x))

    return RichResult(payload={
        "fetal": rebuild(frow),
        "maternal": rebuild(mrow),
        "fetalrow": frow,
        "maternalrow": mrow,
        "peaks": peaks,
        "W": W,
        "H": H,
        "error": err,
        "method": "single-channel fetal ECG extraction by NMF of the STFT "
                  "magnitude with activation-peak selection, Rangayyan "
                  "Biomedical Signal Analysis 3rd ed. Section 9.11, "
                  "eqs. (9.88)-(9.91)",
    })


rangayyan_fetal_ecg_single = fecgnmf  # pre-policy spelling


# -- rgecgnl: Normal vs. ectopic ECG beat classification.
def pvclindf(rr, ff, train=None):
    """Classify ECG beats as normal or PVC with a linear discriminant on [RR, FF].

    Why: a premature ventricular contraction has both a shorter preceding RR
    interval and a more complex waveshape than a normal beat of the same
    subject, so those two numbers alone separate the classes.  The form factor
    of eq. (5.26) makes the second, qualitative half of the clinical rule in
    Section 10.2.2 into a measurable quantity.  Rangayyan, *Biomedical Signal
    Analysis*, 3rd ed., Section 10.11.1.

    With no training data the published decision function is used, eq.
    (10.131):

        RR - 5.56 FF + 11.44  > 0  -> normal beat,
                              <= 0 -> PVC,

    which the book derived as the normal bisector of the segment joining the
    prototype vectors [RR, FF] = [0.66, 1.58] for normal beats and
    [0.45, 2.74] for PVCs, and which classified the whole training set
    correctly.  Given ``train``, the same construction is redone on the
    supplied data: class prototypes, then their perpendicular bisector, the
    linear decision function of Section 10.4.1.

    Parameters
    ----------
    rr : sequence
        RR interval of each beat, in seconds.
    ff : sequence
        Form factor of each beat, eq. (5.26).
    train : tuple, optional
        ``(rr, ff, labels)`` with labels 0 for normal and 1 for PVC, to derive
        the decision function instead of using the published coefficients.

    Returns
    -------
    RichResult
        Keys ``labels``, ``discriminant``, ``coefficients``, ``prototypes``,
        ``method``.
    """
    rr = _bxvec(rr, "rr")
    ff = _bxvec(ff, "ff")
    if len(rr) != len(ff):
        raise ValueError("rr and ff must have the same length")

    proto = None
    if train is None:
        a, b, c = 1.0, -5.56, 11.44
        proto = {"normal": [0.66, 1.58], "pvc": [0.45, 2.74]}
        src = "published coefficients of eq. (10.131)"
    else:
        try:
            trr, tff, tlab = train
        except (TypeError, ValueError):
            raise ValueError("train must be a (rr, ff, labels) triple")
        trr = _bxvec(trr, "train rr")
        tff = _bxvec(tff, "train ff")
        tlab = [int(t) for t in _bxvec(tlab, "train labels")]
        if not (len(trr) == len(tff) == len(tlab)):
            raise ValueError("training rr, ff and labels must have equal length")
        if set(tlab) != {0, 1}:
            raise ValueError("training labels must contain both 0 (normal) and 1 (PVC)")
        p0 = [_bxmean([trr[i] for i in range(len(tlab)) if tlab[i] == 0]),
              _bxmean([tff[i] for i in range(len(tlab)) if tlab[i] == 0])]
        p1 = [_bxmean([trr[i] for i in range(len(tlab)) if tlab[i] == 1]),
              _bxmean([tff[i] for i in range(len(tlab)) if tlab[i] == 1])]
        d = [p1[0] - p0[0], p1[1] - p0[1]]
        if abs(d[0]) < 1e-12 and abs(d[1]) < 1e-12:
            raise ValueError("the two class prototypes coincide")
        mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])]
        a, b = -d[0], -d[1]
        c = d[0] * mid[0] + d[1] * mid[1]
        proto = {"normal": p0, "pvc": p1}
        src = "coefficients derived from the supplied training set"

    disc = [a * rr[i] + b * ff[i] + c for i in range(len(rr))]
    labs = [0 if t > 0.0 else 1 for t in disc]
    return RichResult(payload={
        "labels": labs,
        "discriminant": disc,
        "coefficients": {"rr": a, "ff": b, "constant": c},
        "prototypes": proto,
        "source": src,
        "method": "linear discriminant on [RR interval, form factor] for "
                  "normal vs. ectopic beats, Rangayyan Biomedical Signal "
                  "Analysis 3rd ed. Section 10.11.1, eq. (10.131)",
    })


rangayyan_ecg_normal_ectopic = pvclindf  # pre-policy spelling


# -- rgeegb: EEG rhythm band classification (delta/theta/alpha/beta/gamma).
def eegbands(x, fs, bands=None):
    """Fractional power of an EEG segment in the named rhythm bands.

    Why: the EEG is described clinically by its rhythms, and the question a
    reader actually asks -- is there an alpha rhythm? -- is answered by the
    fraction of signal power falling in that band.  Rangayyan, *Biomedical
    Signal Analysis*, 3rd ed., Section 1.2.6 defines the bands and Section
    10.2.3 makes the fractional-power test explicit.

    Band limits are exactly those of Section 1.2.6:
    delta 0.5 <= f < 4 Hz; theta 4 <= f < 8 Hz; alpha 8 <= f <= 13 Hz;
    beta f > 13 Hz; and, defined separately in that section, gamma 30 - 80 Hz.
    The fraction of power in a band f1:f2 is eq. (6.44), the sum of |X(k)|^2
    over the band divided by the total.

    Parameters
    ----------
    x : sequence
        EEG segment.
    fs : float
        Sampling rate in Hz.
    bands : dict, optional
        Override the band limits, as ``{name: (f1, f2)}`` in Hz.

    Returns
    -------
    RichResult
        Keys ``power``, ``fraction``, ``dominant``, ``totalpower``,
        ``frequencies``, ``method``.
    """
    x = _bxvec(x, "x")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be a positive sampling rate in Hz")
    if len(x) < 8:
        raise ValueError("need at least eight samples to estimate a spectrum")
    nyq = fs / 2.0
    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, nyq),
            "gamma": (30.0, min(80.0, nyq)),
        }
    if not isinstance(bands, dict) or not bands:
        raise ValueError("bands must be a non-empty dict of (f1, f2) pairs")

    m = _bxmean(x)
    xc = [t - m for t in x]
    mag = _bxdftmag(xc)
    n = len(xc)
    freqs = [k * fs / n for k in range(len(mag))]
    psd = [t * t for t in mag]
    total = fsum(psd)

    power, frac = {}, {}
    for name, lim in bands.items():
        try:
            f1, f2 = float(lim[0]), float(lim[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError("band %r must be an (f1, f2) pair" % (name,))
        if f1 < 0.0 or f2 <= f1:
            raise ValueError("band %r must satisfy 0 <= f1 < f2" % (name,))
        p = fsum(psd[k] for k in range(len(psd)) if f1 <= freqs[k] <= f2)
        power[name] = p
        frac[name] = p / total if total > 0.0 else 0.0

    dom = max(frac, key=lambda t: frac[t]) if total > 0.0 else None
    return RichResult(payload={
        "power": power,
        "fraction": frac,
        "dominant": dom,
        "totalpower": total,
        "frequencies": freqs,
        "bands": {k: (float(v[0]), float(v[1])) for k, v in bands.items()},
        "method": "fractional power in the EEG rhythm bands, band limits from "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 1.2.6, "
                  "fraction by eq. (6.44) as used in Section 10.2.3",
    })


rangayyan_eeg_rhythms = eegbands  # pre-policy spelling


# -- rgelbow: Elbow method for k-means cluster count selection.
def elbow(X, kmax=8, kmin=1):
    """Elbow method for choosing the number of clusters.

        WCSS(k) = sum_k sum_{x in C_k} ||x - mu_k||^2

    WCSS falls monotonically with k and reaches zero at k = n, so it
    cannot be minimized -- the choice is the KNEE, where the fall stops
    being worth the extra cluster.  The knee is located here as the point
    of maximum distance from the chord joining the first and last points
    of the curve, which is a definite rule rather than an eye judgement.

    It remains a heuristic: on data with no cluster structure at all the
    curve is smooth and the "knee" is wherever the arithmetic puts it.
    The monotonicity of the curve is checked and returned, since a rise
    would mean the k-means runs landed in bad local minima.
    """
    Xs = _mat(X)
    n = len(Xs)
    lo, hi = int(kmin), int(kmax)
    if lo < 1:
        raise ValueError("kmin must be at least 1")
    if hi > n:
        raise ValueError("kmax exceeds the number of patterns")
    if hi <= lo:
        raise ValueError("kmax must exceed kmin")
    ks, wcss = [], []
    for k in range(lo, hi + 1):
        r = kmeans(Xs, k)
        ks.append(k)
        wcss.append(r["wcss"])
    mono = all(b <= a + 1e-9 for a, b in zip(wcss, wcss[1:]))
    x1, y1 = float(ks[0]), wcss[0]
    x2, y2 = float(ks[-1]), wcss[-1]
    den = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if den <= 0:
        knee = ks[0]
    else:
        dists = [abs((y2 - y1) * kx - (x2 - x1) * ky + x2 * y1 - y2 * x1)
                 / den for kx, ky in zip([float(v) for v in ks], wcss)]
        knee = ks[max(range(len(ks)), key=lambda i: dists[i])]
    return RichResult(payload={
        "k": ks, "wcss": wcss, "knee": knee,
        "monotonic": mono,
        "wcss_cannot_be_minimized": True,
        "heuristic_only": True,
        "method": "elbow criterion on the k-means WCSS; Rangayyan "
                  "(2024) Section 10.5.1"})


rangayyan_kmeans_elbow = elbow  # pre-policy spelling


# -- rgepiksv: Epileptic seizure detection using K-SVD dictionary learning.
def seizdict(signals, labels, iterations=7, atoms=None, test=None):
    """Learn a signal-derived dictionary and use it to detect epileptic seizures.

    Why: the EEG is nonstationary, so a fixed basis represents seizure and
    nonseizure segments equally badly.  Learning the atoms from the recordings
    themselves gives a small dictionary whose atoms carry properties unique to
    each class, and the projection coefficients and the reconstruction error
    against that dictionary are then the seizure-detection features.
    Rangayyan, *Biomedical Signal Analysis*, 3rd ed., Section 9.8, following
    the dictionary-learning framework of Section 9.5.

    Implements Algorithm 9.2 verbatim: for each training signal, repeatedly
    take the projection coefficient alpha_m = <x, psi_m> against every atom of
    that signal's raw dictionary, add the atom of largest |alpha_m| to the
    trained dictionary if it is not already there, remove it from the raw
    dictionary, and replace x by its residue.  The book runs I = 1 to 7.

    Classification uses the features named in Section 9.8 -- the projection
    coefficient vector and the reconstruction error -- assigning each signal to
    the class whose mean training feature vector is nearer.

    Parameters
    ----------
    signals : sequence of sequences
        Training signals, one per row, all the same length.
    labels : sequence
        Class code per training signal, e.g. 0 nonseizure and 1 seizure.
    iterations : int
        Number of passes I; the book uses 7.
    atoms : sequence of sequences, optional
        A raw dictionary shared by all signals.  By default each training
        signal contributes its own L2-normalised half-overlapping segments,
        which stands in for the DWT or EMD components of the book.
    test : sequence of sequences, optional
        Further signals to classify with the learned dictionary.

    Returns
    -------
    RichResult
        Keys ``dictionary``, ``coefficients``, ``error``, ``predictions``,
        ``testclass``, ``isseizure``, ``accuracy``, ``method``.
    """
    S = _bxmat(signals, "signals")
    y = [int(t) for t in _bxvec(labels, "labels")]
    if len(S) != len(y):
        raise ValueError("signals and labels must have the same length")
    if len(set(y)) < 2:
        raise ValueError("labels must contain at least two classes")
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("iterations must be a positive integer")
    n = len(S[0])

    def rawdict(sig):
        if atoms is not None:
            A = _bxmat(atoms, "atoms")
            if any(len(a) != n for a in A):
                raise ValueError("atoms must have the same length as the signals")
            return [list(a) for a in A]
        seg = max(4, n // 4)
        out = []
        for st in range(0, n - seg + 1, max(1, seg // 2)):
            a = [0.0] * n
            for i in range(seg):
                a[st + i] = sig[st + i]
            nr = _bxnrm(a)
            if nr > 1e-12:
                out.append([t / nr for t in a])
        return out

    trained = []
    for s in range(len(S)):
        raw = rawdict(S[s])
        x = list(S[s])
        for _ in range(iterations):
            if not raw:
                break
            best, bv = -1, -1.0
            for j in range(len(raw)):
                v = abs(_bxdot(x, raw[j]))
                if v > bv:
                    best, bv = j, v
            if best < 0:
                break
            psi = raw.pop(best)
            if not any(all(abs(psi[i] - d[i]) < 1e-12 for i in range(n))
                       for d in trained):
                trained.append(psi)
            a = _bxdot(x, psi)
            x = [x[i] - a * psi[i] for i in range(n)]
    if not trained:
        raise ValueError("dictionary learning produced no atoms; "
                         "check that the signals are not all zero")

    def feats(sig):
        co = [_bxdot(sig, d) for d in trained]
        rec = [fsum(co[j] * trained[j][i] for j in range(len(trained)))
               for i in range(n)]
        err = _bxnrm([sig[i] - rec[i] for i in range(n)])
        return co + [err], co, err

    Ftr = [feats(row) for row in S]
    classes = sorted(set(y))
    cent = {}
    for c in classes:
        rows = [Ftr[i][0] for i in range(len(y)) if y[i] == c]
        cent[c] = [fsum(r[j] for r in rows) / len(rows) for j in range(len(rows[0]))]

    def assign(v):
        return min(classes, key=lambda c: fsum((v[j] - cent[c][j]) ** 2
                                               for j in range(len(v))))

    pred = [assign(Ftr[i][0]) for i in range(len(y))]
    acc = fsum(1.0 for a, b in zip(y, pred) if a == b) / len(y)

    tcls = None
    if test is not None:
        T = _bxmat(test, "test")
        if any(len(r) != n for r in T):
            raise ValueError("test signals must have the same length as training signals")
        tcls = [assign(feats(r)[0]) for r in T]

    return RichResult(payload={
        "dictionary": trained,
        "coefficients": [Ftr[i][1] for i in range(len(y))],
        "error": [Ftr[i][2] for i in range(len(y))],
        "predictions": pred,
        "isseizure": [t == max(classes) for t in pred],
        "testclass": tcls,
        "accuracy": acc,
        "method": "signal-derived dictionary learning (Algorithm 9.2) with "
                  "projection-coefficient and reconstruction-error features "
                  "for seizure detection, Rangayyan Biomedical Signal Analysis "
                  "3rd ed. Section 9.8",
    })


rangayyan_epilepsy_ksvd = seizdict  # pre-policy spelling


# -- rgerrbd: Bhattacharyya bound on Bayes classification error.
def errbound(p1, p2, db):
    """Chernoff-Bhattacharyya bound on the Bayes error.

        P_e <= sqrt(P1 P2) exp(-D_B)

    NOT FROM THIS BOOK.  Rangayyan (2024) does not give a Bhattacharyya
    error bound; this is the standard Kailath bound and is kept because
    the name was already exposed.  It pairs with ``gaussoverlap``, not with the
    book's ``divergence`` -- the divergence does NOT bound the error this
    way, and substituting it here would give a number that looks like a
    bound and is not one.

    The bound is on the error of the OPTIMAL (Bayes) classifier, so it is
    a floor no real classifier can beat, not a promise any will reach it.
    """
    a, b = float(p1), float(p2)
    if a < 0 or b < 0:
        raise ValueError("prior probabilities cannot be negative")
    if abs(a + b - 1.0) > 1e-9:
        raise ValueError("the two priors must sum to 1, got %g" % (a + b))
    d = float(db)
    if d < 0:
        raise ValueError("the Bhattacharyya distance cannot be negative")
    bound = sqrt(a * b) * exp(-d)
    return RichResult(payload={
        "bound": bound, "priors": [a, b], "bhattacharyya": d,
        "tightest_at_equal_priors": abs(a - b) < 1e-12,
        "bounds_the_optimal_classifier_not_yours": True,
        "not_from_this_book": True,
        "pairs_with_the_overlap_not_with_divergence": True,
        "reference": "Kailath T. The divergence and Bhattacharyya "
                     "distance measures in signal selection. IEEE "
                     "Transactions on Communication Technology "
                     "15(1):52-60, February 1967, "
                     "doi:10.1109/TCOM.1967.1089532.",
        "method": "Kailath's Bhattacharyya bound; not given in "
                  "Rangayyan (2024)"})


rangayyan_bayes_error_bound = errbound  # pre-policy spelling


# -- rgfish: Fisher's criterion for feature separability.
def fishcrit(x1, x2):
    """Fisher's criterion for a scalar feature.

        J = (m1 - m2)^2 / (s1^2 + s2^2)

    The squared mean separation over the summed variances.  It is close
    kin to the book's normalized distance of eq. (10.112) but NOT the
    same measure: eq. (10.112) divides |m1 - m2| by (s1 + s2), this
    divides the square by the sum of squares.  They rank features
    identically only when the two dispersions are equal, so a feature
    ranking built with one and reported as the other is wrong wherever
    the classes have unequal spread.  Both are returned.
    """
    a, b = aslist(x1), aslist(x2)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each class needs at least two samples")
    m1 = fsum(a) / len(a)
    m2 = fsum(b) / len(b)
    v1 = fsum((v - m1) ** 2 for v in a) / (len(a) - 1)
    v2 = fsum((v - m2) ** 2 for v in b) / (len(b) - 1)
    den = v1 + v2
    if den <= 0:
        raise ValueError("both classes have zero variance; the criterion "
                         "is undefined")
    s1, s2 = sqrt(v1), sqrt(v2)
    dn = abs(m1 - m2) / (s1 + s2) if (s1 + s2) > 0 else float("inf")
    return RichResult(payload={
        "j": (m1 - m2) ** 2 / den, "means": [m1, m2],
        "variances": [v1, v2], "normalized_distance": dn,
        "agrees_with_eq_10_112_ranking_only_for_equal_spread":
            abs(s1 - s2) < 1e-12,
        "is_not_eq_10_112": True,
        "method": "Fisher's criterion; compare Rangayyan (2024) "
                  "eq. (10.112)"})


rangayyan_fisher_criterion = fishcrit  # pre-policy spelling


# -- rgfld: Fisher linear discriminant analysis (LDA).
def fishlda(X, y):
    """Fisher linear discriminant analysis, Section 10.4.2.

        w = S_W^-1 (m1 - m2)

    the direction that maximizes the ratio of between-class to
    within-class scatter of the PROJECTED data.  Two classes only: the
    two-class Fisher direction is a single vector, and the multiclass
    generalization is a different construction with up to k-1 directions.

    The projection is a dimension reduction to ONE number, chosen for
    separation and not for reconstruction, so it discards everything
    orthogonal to w -- unlike a principal component, it is not meant to
    represent the data, only to separate it.
    """
    Xs = _mat(X)
    ys = list(y)
    if len(Xs) != len(ys):
        raise ValueError("X and y must have the same number of rows")
    p = len(Xs[0])
    order, grp = _groups(Xs, ys)
    if len(order) != 2:
        raise ValueError("Fisher's linear discriminant as stated is a "
                         "two-class method; got %d classes"
                         % len(order))
    a, b = grp[order[0]], grp[order[1]]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each class needs at least two samples")
    m1, m2 = _colmeans(a), _colmeans(b)
    SW = _scatter(a, m1)
    s2 = _scatter(b, m2)
    for i in range(p):
        for j in range(p):
            SW[i][j] += s2[i][j]
    diff = [m1[i] - m2[i] for i in range(p)]
    Wi = _inv(SW)
    w = [fsum(Wi[i][j] * diff[j] for j in range(p)) for i in range(p)]
    proj_a = [fsum(w[j] * r[j] for j in range(p)) for r in a]
    proj_b = [fsum(w[j] * r[j] for j in range(p)) for r in b]
    ma = fsum(proj_a) / len(proj_a)
    mb = fsum(proj_b) / len(proj_b)
    va = fsum((v - ma) ** 2 for v in proj_a)
    vb = fsum((v - mb) ** 2 for v in proj_b)
    thr = 0.5 * (ma + mb)
    return RichResult(payload={
        "w": w, "threshold": thr, "classes": order,
        "means": [m1, m2], "s_within": SW,
        "projected": {order[0]: proj_a, order[1]: proj_b},
        "projected_means": [ma, mb],
        "criterion": ((ma - mb) ** 2 / (va + vb)) if (va + vb) > 0
        else float("inf"),
        "two_class_only": True,
        "not_a_reconstruction_basis": True,
        "method": "Rangayyan (2024) Section 10.4.2 (Fisher LDA)"})


rangayyan_fisher_lda = fishlda  # pre-policy spelling


# -- rghier: Hierarchical agglomerative clustering.
def hclust(X, linkage="single", k=None):
    """Hierarchical agglomerative clustering, Section 10.5.1.

    Start with every pattern its own cluster and repeatedly merge the two
    closest, by

      "single"    the nearest pair of members  -- chains, so it will
                  string distant clusters together through a bridge of
                  intermediate points
      "complete"  the furthest pair           -- compact, and splits
                  elongated clusters
      "average"   the mean over all pairs     -- between the two

    The choice of linkage is not cosmetic: on the same data single and
    complete linkage routinely give different partitions, which is why
    the merge history is returned in full rather than only a labelling.

    Cutting the tree at ``k`` clusters gives labels; without it only the
    dendrogram history is returned.
    """
    Xs = _mat(X)
    n = len(Xs)
    if n < 2:
        raise ValueError("need at least two patterns")
    if linkage not in ("single", "complete", "average"):
        raise ValueError("linkage must be 'single', 'complete' or "
                         "'average'")
    p = len(Xs[0])

    def d2(a, b):
        return sqrt(fsum((a[i] - b[i]) ** 2 for i in range(p)))

    groups = {i: [i] for i in range(n)}
    history = []
    while len(groups) > 1:
        keys = sorted(groups)
        best = None
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ga, gb = groups[keys[a]], groups[keys[b]]
                ds = [d2(Xs[i], Xs[j]) for i in ga for j in gb]
                if linkage == "single":
                    dd = min(ds)
                elif linkage == "complete":
                    dd = max(ds)
                else:
                    dd = fsum(ds) / len(ds)
                if best is None or dd < best[0]:
                    best = (dd, keys[a], keys[b])
        dd, ka, kb = best
        history.append({"merged": (ka, kb), "distance": dd,
                        "size": len(groups[ka]) + len(groups[kb]),
                        "n_clusters_after": len(groups) - 1})
        groups[ka] = groups[ka] + groups[kb]
        del groups[kb]
    labels = None
    if k is not None:
        kk = int(k)
        if not 1 <= kk <= n:
            raise ValueError("k must lie in 1..n")
        g = {i: [i] for i in range(n)}
        for step in history:
            if len(g) == kk:
                break
            ka, kb = step["merged"]
            g[ka] = g[ka] + g[kb]
            del g[kb]
        labels = [0] * n
        for c, key in enumerate(sorted(g)):
            for i in g[key]:
                labels[i] = c
    return RichResult(payload={
        "history": history, "labels": labels, "linkage": linkage,
        "n": n, "k": k,
        "merge_distances": [h["distance"] for h in history],
        "monotonic_merges": all(
            b >= a - 1e-12 for a, b in zip(
                [h["distance"] for h in history],
                [h["distance"] for h in history][1:])),
        "single_linkage_chains": linkage == "single",
        "linkage_changes_the_partition": True,
        "method": "Rangayyan (2024) Section 10.5.1 (cluster seeking)"})


rangayyan_hierarchical_clust = hclust  # pre-policy spelling


# -- rgica: FastICA algorithm for independent component analysis.
def icafix(X, ncomp=None, maxiter=200, tol=1e-8, seed=1):
    """Independent component analysis by the fixed-point (FastICA) algorithm.

    Why: PCA can only make components uncorrelated, which equals independence
    only when the sources are Gaussian.  Since a linear mixture tends towards
    a Gaussian by the central limit theorem, driving the estimated sources
    *away* from Gaussianity is what actually unmixes them.  Rangayyan,
    *Biomedical Signal Analysis*, 3rd ed., Section 9.7.2 states the model
    y(n) = M x(n), eq. (9.43), the unmixing xtilde(n) = W y(n), eq. (9.44),
    and the kurtosis-based objective, and names FastICA as the efficient
    implementation; it gives only the generic gradient rule (9.45).

    The fixed-point update used here is that of Hyvarinen and Oja,
    "Independent component analysis: Algorithms and applications", Neural
    Networks 13, 2000, listed as reference [50] of Section 9.7.2:

        w <- E{y g(w'y)} - E{g'(w'y)} w,   g(u) = tanh(u),

    applied one component at a time with Gram-Schmidt deflation, on data first
    centred and whitened by the PCA of Section 9.7.1.

    Parameters
    ----------
    X : sequence of sequences
        Observed mixtures, one channel per row, samples along the row.
    ncomp : int, optional
        Number of components; defaults to the number of channels.
    maxiter : int
        Fixed-point iterations per component.
    tol : float
        Convergence tolerance on the direction change.
    seed : int
        Seed for the deterministic starting directions.

    Returns
    -------
    RichResult
        Keys ``sources``, ``unmixing``, ``mixing``, ``whitening``, ``mean``,
        ``iterations``, ``method``.
    """
    Y = _bxmat(X, "X")
    K, T = len(Y), len(Y[0])
    if T < 4:
        raise ValueError("need at least four samples per channel")
    L = K if ncomp is None else int(ncomp)
    if not (1 <= L <= K):
        raise ValueError("ncomp must satisfy 1 <= ncomp <= number of channels")
    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    mu = [_bxmean(r) for r in Y]
    Yc = [[Y[i][t] - mu[i] for t in range(T)] for i in range(K)]
    C = [[fsum(Yc[i][t] * Yc[j][t] for t in range(T)) / T for j in range(K)]
         for i in range(K)]
    vals, vecs = _bxjacobi(C)
    if vals[L - 1] <= 1e-14:
        raise ValueError("the mixture covariance is rank deficient for %d components" % L)
    Wh = [[vecs[j][k] / sqrt(vals[k]) for j in range(K)] for k in range(L)]
    Z = _bxmm(Wh, Yc)

    u = _bxrng(seed)
    Wm, iters = [], []
    for c in range(L):
        w = [u() - 0.5 for _ in range(L)]
        nr = _bxnrm(w)
        w = [t / nr for t in w] if nr > 1e-12 else [1.0 if i == c else 0.0
                                                    for i in range(L)]
        it = 0
        for it in range(1, maxiter + 1):
            g, gp = [0.0] * L, 0.0
            for t in range(T):
                s = fsum(w[i] * Z[i][t] for i in range(L))
                gv = tanh(s)
                for i in range(L):
                    g[i] += Z[i][t] * gv
                gp += 1.0 - gv * gv
            wn = [g[i] / T - (gp / T) * w[i] for i in range(L)]
            for prev in Wm:
                d = _bxdot(wn, prev)
                wn = [wn[i] - d * prev[i] for i in range(L)]
            nr = _bxnrm(wn)
            if nr <= 1e-12:
                wn = [1.0 if i == c else 0.0 for i in range(L)]
                nr = 1.0
            wn = [t / nr for t in wn]
            if abs(abs(_bxdot(wn, w)) - 1.0) < tol:
                w = wn
                break
            w = wn
        Wm.append(w)
        iters.append(it)

    S = _bxmm(Wm, Z)
    W = _bxmm(Wm, Wh)
    Wt = _bxtr(W)
    G = _bxmm(W, Wt)
    for i in range(L):
        G[i][i] += 1e-12
    A = [[fsum(Wt[i][k] * _bxsolve(G, [1.0 if r == j else 0.0
                                       for r in range(L)])[k] for k in range(L))
          for j in range(L)] for i in range(K)]
    return RichResult(payload={
        "sources": S,
        "unmixing": W,
        "mixing": A,
        "whitening": Wh,
        "mean": mu,
        "iterations": iters,
        "method": "FastICA fixed-point ICA with the tanh nonlinearity; model "
                  "and unmixing per Rangayyan Biomedical Signal Analysis 3rd "
                  "ed. Section 9.7.2, eqs. (9.43)-(9.44); update rule from "
                  "Hyvarinen and Oja, Neural Networks 13, 2000 (ref. [50] there)",
    })


rangayyan_fastica = icafix  # pre-policy spelling


# -- rgicaart: EEG artifact removal via ICA (eye blink, muscle, ECG).
def icaclean(X, ncomp=None, kurtosis=3.0, drop=None, maxiter=200, seed=1):
    """Remove artifact components from multichannel EEG by ICA.

    Why: eye blinks, muscle activity and the ECG contaminate the EEG through
    separate physical sources, so they appear as separate independent
    components of the mixture rather than as a separable frequency band.
    Once the components are estimated, an artifact is suppressed by zeroing
    its component and back-projecting the rest through the mixing matrix,
    which leaves the neural channels intact.  Rangayyan, *Biomedical Signal
    Analysis*, 3rd ed., Section 9.7.2 (the model and the kurtosis criterion),
    with ICA-based EEG artifact reduction noted in Section 9.12.

    Components are ranked by the kurtosis excess K' = K - 3 of eq. (3.5) and
    the note following it: K' is zero for a Gaussian, strongly positive for
    the peaked, heavy-tailed waveform of a blink or a QRS spike.  Components
    whose |K'| exceeds ``kurtosis`` are treated as artifacts unless ``drop``
    names the components explicitly.

    Parameters
    ----------
    X : sequence of sequences
        EEG channels, one per row.
    ncomp : int, optional
        Number of components to estimate.
    kurtosis : float
        Threshold on |K'| for automatic artifact flagging.
    drop : sequence of int, optional
        Component indices to remove, overriding the automatic rule.
    maxiter : int
        FastICA iteration budget.
    seed : int
        Seed for the deterministic starting directions.

    Returns
    -------
    RichResult
        Keys ``clean``, ``components``, ``kurtosis``, ``artifacts``,
        ``mixing``, ``removedpower``, ``method``.
    """
    ica = icafix(X, ncomp=ncomp, maxiter=maxiter, seed=seed)
    S = ica["sources"]
    A = ica["mixing"]
    mu = ica["mean"]
    L, T = len(S), len(S[0])

    kv = [_bxkurt(S[c]) for c in range(L)]
    if drop is None:
        thr = float(kurtosis)
        if thr < 0.0:
            raise ValueError("kurtosis threshold must be nonnegative")
        art = [c for c in range(L) if abs(kv[c]) > thr]
    else:
        art = sorted({int(t) for t in drop})
        if any(c < 0 or c >= L for c in art):
            raise ValueError("drop indices must lie in [0, %d)" % L)

    Sk = [[0.0] * T if c in art else list(S[c]) for c in range(L)]
    rec = _bxmm(A, Sk)
    clean = [[rec[i][t] + mu[i] for t in range(T)] for i in range(len(A))]
    orig = fsum(fsum(t * t for t in S[c]) for c in range(L))
    gone = fsum(fsum(t * t for t in S[c]) for c in art)
    return RichResult(payload={
        "clean": clean,
        "components": S,
        "kurtosis": kv,
        "artifacts": art,
        "mixing": A,
        "removedpower": gone / orig if orig > 0.0 else 0.0,
        "method": "ICA artifact removal by zeroing high-kurtosis components and "
                  "back-projection, Rangayyan Biomedical Signal Analysis 3rd "
                  "ed. Section 9.7.2 with the kurtosis excess of eq. (3.5)",
    })


rangayyan_ica_artifact = icaclean  # pre-policy spelling


# -- rginf: Infomax ICA algorithm (Bell-Sejnowski).
def infomax(X, ncomp=None, eta=0.05, maxiter=300, tol=1e-8, seed=1):
    """Blind source separation by the Infomax rule with the natural gradient.

    Why: FastICA maximises non-Gaussianity one direction at a time; Infomax
    instead maximises the joint entropy of a nonlinearly squashed output, which
    updates the whole unmixing matrix at once and degrades gracefully when the
    sources are not exactly independent.  Both estimate the same unmixing
    matrix W of Rangayyan, *Biomedical Signal Analysis*, 3rd ed., eq. (9.44),
    xtilde(n) = W y(n), for the mixture model of eq. (9.43).

    Not from Rangayyan: Section 9.7.2 presents only the generic gradient rule
    (9.45), Wtilde_(n+1) = Wtilde_n - mu grad F, and does not give the Infomax
    update.  Primary sources are Bell and Sejnowski, "An information-
    maximization approach to blind separation and blind deconvolution", Neural
    Computation 7(6):1129-1159, 1995, and Amari, Cichocki and Yang, Advances in
    Neural Information Processing Systems 8:757-763, 1996, for the natural
    gradient form used here:

        u = W z,   phi(u) = 1 - 2 sigmoid(u),
        Delta W = eta (I + phi(u) u') W.

    Data are centred and whitened first, exactly as for ``icafix``.

    The logistic nonlinearity above matches super-Gaussian (sparse, spiky)
    sources, which is what most biomedical artifacts are.  On sub-Gaussian
    sources -- a square wave, a uniform process -- this rule separates only
    partially; the extended Infomax of Lee, Girolami and Sejnowski, Neural
    Computation 11(2):417-441, 1999, switches the nonlinearity by the sign of
    the kurtosis and is the remedy.  Use ``icafix`` when the source kurtosis
    sign is unknown.

    Parameters
    ----------
    X : sequence of sequences
        Observed mixtures, one channel per row.
    ncomp : int, optional
        Number of components; defaults to the number of channels.
    eta : float
        Learning rate.
    maxiter : int
        Number of passes over the data.
    tol : float
        Stop when the largest weight change falls below this.
    seed : int
        Seed for the deterministic initialisation.

    Returns
    -------
    RichResult
        Keys ``sources``, ``unmixing``, ``whitening``, ``mean``,
        ``iterations``, ``change``, ``method``.
    """
    Y = _bxmat(X, "X")
    K, T = len(Y), len(Y[0])
    if T < 4:
        raise ValueError("need at least four samples per channel")
    L = K if ncomp is None else int(ncomp)
    if not (1 <= L <= K):
        raise ValueError("ncomp must satisfy 1 <= ncomp <= number of channels")
    eta = float(eta)
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must lie in (0, 1]")
    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    mu = [_bxmean(r) for r in Y]
    Yc = [[Y[i][t] - mu[i] for t in range(T)] for i in range(K)]
    C = [[fsum(Yc[i][t] * Yc[j][t] for t in range(T)) / T for j in range(K)]
         for i in range(K)]
    vals, vecs = _bxjacobi(C)
    if vals[L - 1] <= 1e-14:
        raise ValueError("the mixture covariance is rank deficient for %d components" % L)
    Wh = [[vecs[j][k] / sqrt(vals[k]) for j in range(K)] for k in range(L)]
    Z = _bxmm(Wh, Yc)

    u = _bxrng(seed)
    W = [[(1.0 if i == j else 0.0) + 0.01 * (u() - 0.5) for j in range(L)]
         for i in range(L)]

    def sig(b):
        if b < -700.0:
            return 0.0
        if b > 700.0:
            return 1.0
        return 1.0 / (1.0 + exp(-b))

    it, chg = 0, float("nan")
    for it in range(1, maxiter + 1):
        U = _bxmm(W, Z)
        P = [[1.0 - 2.0 * sig(U[i][t]) for t in range(T)] for i in range(L)]
        M = [[(1.0 if i == j else 0.0)
              + fsum(P[i][t] * U[j][t] for t in range(T)) / T
              for j in range(L)] for i in range(L)]
        D = _bxmm(M, W)
        chg = max(abs(D[i][j]) for i in range(L) for j in range(L)) * eta
        for i in range(L):
            for j in range(L):
                W[i][j] += eta * D[i][j]
        nrm = max(abs(W[i][j]) for i in range(L) for j in range(L))
        if not isfinite(nrm) or nrm > 1e8:
            raise ValueError("Infomax diverged; reduce eta")
        if chg <= tol:
            break

    S = _bxmm(W, Z)
    return RichResult(payload={
        "sources": S,
        "unmixing": _bxmm(W, Wh),
        "whitening": Wh,
        "mean": mu,
        "iterations": it,
        "change": chg,
        "method": "Infomax ICA with the natural gradient; unmixing model per "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. eq. (9.44), "
                  "update from Bell and Sejnowski, Neural Computation "
                  "7(6):1129-1159, 1995 and Amari, Cichocki and Yang, NIPS "
                  "8:757-763, 1996 (not covered by Rangayyan)",
    })


rangayyan_infomax_ica = infomax  # pre-policy spelling


# -- rgkfcv: K-fold cross-validation.
def kfoldcv(X, y, k=5, classifier=None, stratified=True):
    """K-fold cross-validation, Section 10.10.3.

        CV = (1/K) sum_k error on held-out fold k

    The book's point in Section 10.10.3 is that the training and test
    steps must use SEPARATE data: an error rate measured on the samples
    that trained the classifier is optimistic, sometimes wildly so, and
    with enough free parameters it reaches zero while the classifier
    generalizes not at all.

    Folds are stratified by default, keeping each class's proportion in
    every fold.  Unstratified folds on unbalanced data can leave a class
    absent from a training fold entirely, which does not measure
    generalization so much as luck.

    ``classifier`` is a callable (X_train, y_train, x) -> label; the
    default is 1-NN.
    """
    Xs = _mat(X)
    ys = list(y)
    n = len(Xs)
    if n != len(ys):
        raise ValueError("X and y must have the same number of rows")
    kk = int(k)
    if not 2 <= kk <= n:
        raise ValueError("k must lie in 2..n")
    if classifier is None:
        def classifier(Xt, yt, q):
            return knn(Xt, yt, q, k=1)["assigned"]
    if stratified:
        order, _ = _groups(Xs, ys)
        by = {lab: [i for i in range(n) if ys[i] == lab] for lab in order}
        folds = [[] for _ in range(kk)]
        c = 0
        for lab in order:
            for i in by[lab]:
                folds[c % kk].append(i)
                c += 1
    else:
        folds = [[i for i in range(n) if i % kk == f] for f in range(kk)]
    errors, per_fold = 0, []
    for f in range(kk):
        test = folds[f]
        if not test:
            continue
        tr = [i for i in range(n) if i not in set(test)]
        if len(set(ys[i] for i in tr)) < 2:
            raise ValueError("fold %d leaves fewer than two classes in "
                             "the training set; use stratified folds or "
                             "a smaller k" % f)
        Xt = [Xs[i] for i in tr]
        yt = [ys[i] for i in tr]
        e = sum(1 for i in test if classifier(Xt, yt, Xs[i]) != ys[i])
        errors += e
        per_fold.append({"fold": f, "n": len(test), "errors": e,
                         "error_rate": e / len(test)})
    rate = errors / n
    return RichResult(payload={
        "error_rate": rate, "accuracy": 1.0 - rate, "errors": errors,
        "n": n, "k": kk, "per_fold": per_fold,
        "stratified": bool(stratified),
        "train_and_test_must_be_separate": True,
        "method": "Rangayyan (2024) Section 10.10.3 (training and test "
                  "steps)"})


rangayyan_kfold_cv = kfoldcv  # pre-policy spelling


# -- rgkmns: K-means clustering algorithm.
def kmeans(X, k, maxiter=100, tol=1e-10, init=None):
    """K-means cluster seeking, Section 10.5.1.

    Assign each pattern to the nearest centroid, then move each centroid
    to the mean of the patterns assigned to it, and repeat.  The
    within-cluster sum of squares falls at every step, so the iteration
    always terminates -- at a LOCAL minimum, which depends on where the
    centroids started.  The book's method is unsupervised: it finds
    groups, and whether those groups correspond to the diagnostic classes
    is a separate question the algorithm cannot answer.

    Starting centroids are the first k distinct patterns unless ``init``
    is given, which makes the result reproducible; random starts would
    make the same call return different clusterings.

    An emptied cluster is re-seeded from the point furthest from its
    centroid rather than dropped, so k clusters are always returned.
    """
    Xs = _mat(X)
    n = len(Xs)
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be at least 1")
    if kk > n:
        raise ValueError("k exceeds the number of patterns")
    p = len(Xs[0])
    if init is None:
        seen, cent = [], []
        for r in Xs:
            if r not in seen:
                seen.append(r)
                cent.append(list(r))
            if len(cent) == kk:
                break
        if len(cent) < kk:
            raise ValueError("fewer than k distinct patterns")
    else:
        cent = _mat(init)
        if len(cent) != kk or any(len(r) != p for r in cent):
            raise ValueError("init must be k x p")

    def d2(a, b):
        return fsum((a[i] - b[i]) ** 2 for i in range(p))

    lab = [0] * n
    prev = None
    it = 0
    for it in range(1, int(maxiter) + 1):
        for i in range(n):
            lab[i] = min(range(kk), key=lambda c: d2(Xs[i], cent[c]))
        for c in range(kk):
            rows = [Xs[i] for i in range(n) if lab[i] == c]
            if not rows:
                far = max(range(n), key=lambda i: d2(Xs[i], cent[lab[i]]))
                cent[c] = list(Xs[far])
                lab[far] = c
                rows = [Xs[far]]
            cent[c] = _colmeans(rows)
        wcss = fsum(d2(Xs[i], cent[lab[i]]) for i in range(n))
        if prev is not None and abs(prev - wcss) <= tol:
            break
        prev = wcss
    sizes = [sum(1 for v in lab if v == c) for c in range(kk)]
    return RichResult(payload={
        "labels": lab, "centroids": cent, "wcss": prev, "k": kk,
        "sizes": sizes, "iterations": it,
        "converged": it < int(maxiter),
        "local_minimum_only": True,
        "depends_on_the_starting_centroids": True,
        "unsupervised_groups_need_not_be_the_classes": True,
        "method": "Rangayyan (2024) Section 10.5.1 (cluster seeking)"})


rangayyan_kmeans = kmeans  # pre-policy spelling


# -- rgkneecl: Knee-joint cartilage pathology classification via VAG features.
def vagclass(segments, durations=None, segclass=None, arthro=None):
    """Screen a knee-joint VAG signal for cartilage pathology.

    Why: vibroarthrographic signals are nonstationary, so they are first cut
    into locally stationary segments and each segment is classified on its own.
    The subject-level answer then has to be a rule over the segment verdicts,
    weighted by how much of the recording each segment covers -- a single noisy
    segment must not condemn a normal knee.  Rangayyan, *Biomedical Signal
    Analysis*, 3rd ed., Section 10.12.

    Two features and one decision rule, all from that section:

    * a striking difference between normal and abnormal signals is that
      abnormal ones vary far more in amplitude across a swing cycle; the
      **variance of the segment means** captures it and was used as one of the
      discriminant features;
    * the **two-step rule** of Moussavi et al. as stated in the book: if
      segments spanning more than 90% of the duration are classified normal,
      the subject is normal; if more than 90% of the duration is classified
      abnormal, the subject is abnormal; if more than 10% but less than 90% is
      abnormal, the signal goes to the four-group classifier, and there, if
      more than 10% of the duration is arthroscopically abnormal, the subject
      is abnormal, otherwise normal.

    Parameters
    ----------
    segments : sequence of sequences
        The locally stationary VAG segments, in order.
    durations : sequence, optional
        Duration of each segment; segment lengths are used by default.
    segclass : sequence, optional
        Verdict of the two-group classifier per segment, 0 normal and
        1 abnormal.  Required for the two-step rule.
    arthro : sequence, optional
        Verdict of the four-group classifier per segment, 1 marking
        arthroscopically abnormal, used only when the first step is undecided.

    Returns
    -------
    RichResult
        Keys ``varmeans``, ``segmentmeans``, ``abnormalfraction``,
        ``normalfraction``, ``decision``, ``stage``, ``abnormal``, ``method``.
    """
    S = _bxmat(segments, "segments")
    ns = len(S)
    if durations is None:
        d = [float(len(r)) for r in S]
    else:
        d = _bxvec(durations, "durations")
        if len(d) != ns:
            raise ValueError("durations must have one entry per segment")
        if any(t <= 0.0 for t in d):
            raise ValueError("durations must be positive")
    total = fsum(d)

    smeans = [_bxmean(r) for r in S]
    vms = _bxsd(smeans) ** 2 if ns > 1 else 0.0

    dec, stage, abn = None, None, None
    fabn, fnor = float("nan"), float("nan")
    if segclass is not None:
        sc = [int(t) for t in _bxvec(segclass, "segclass")]
        if len(sc) != ns:
            raise ValueError("segclass must have one entry per segment")
        if any(t not in (0, 1) for t in sc):
            raise ValueError("segclass entries must be 0 (normal) or 1 (abnormal)")
        fabn = fsum(d[i] for i in range(ns) if sc[i] == 1) / total
        fnor = 1.0 - fabn
        if fnor > 0.90:
            dec, stage, abn = "normal", 1, False
        elif fabn > 0.90:
            dec, stage, abn = "abnormal", 1, True
        else:
            if arthro is None:
                dec, stage, abn = "undecided, four-group classifier required", 1, None
            else:
                aa = [int(t) for t in _bxvec(arthro, "arthro")]
                if len(aa) != ns:
                    raise ValueError("arthro must have one entry per segment")
                faa = fsum(d[i] for i in range(ns) if aa[i] == 1) / total
                if faa > 0.10:
                    dec, stage, abn = "abnormal", 2, True
                else:
                    dec, stage, abn = "normal", 2, False

    return RichResult(payload={
        "varmeans": vms,
        "segmentmeans": smeans,
        "abnormalfraction": fabn,
        "normalfraction": fnor,
        "decision": dec,
        "stage": stage,
        "abnormal": abn,
        "durations": d,
        "method": "VAG cartilage-pathology screening: variance of segment means "
                  "plus the two-step 90%/10% duration rule, Rangayyan "
                  "Biomedical Signal Analysis 3rd ed. Section 10.12",
    })


rangayyan_knee_classify = vagclass  # pre-policy spelling


# -- rgknn: K-nearest neighbor (k-NN) classifier.
def knn(X, y, query, k=1, metric="euclidean", C=None):
    """Nearest-neighbour and k-NN rules, eq. (10.29).

        x in C_i  if  D(s_i, x) = min D(s_l, x),  l = 1..N

    and the k-NN rule takes the majority among the k nearest.  The book
    is explicit about why k > 1: with k = 1 "the nearest neighbor may
    happen to be an outlier that is not representative of its class",
    so a single mislabelled or freak training point owns a whole region
    of the feature space.

    ``metric`` may be "euclidean" or "mahalanobis"; the latter needs the
    covariance ``C`` and is the one to use when the features have
    different units or are correlated, since Euclidean distance would
    otherwise be dominated by whichever feature has the largest numbers.

    Ties in the vote are broken toward the class whose voting neighbours
    are nearest, and the fact that a tie occurred is reported rather than
    hidden.
    """
    Xs = _mat(X)
    ys = list(y)
    q = aslist(query)
    if len(Xs) != len(ys):
        raise ValueError("X and y must have the same number of rows")
    if not Xs:
        raise ValueError("need at least one training sample")
    p = len(q)
    if any(len(r) != p for r in Xs):
        raise ValueError("every row of X must match the query length")
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be at least 1")
    if kk > len(Xs):
        raise ValueError("k exceeds the number of training samples")
    if metric not in ("euclidean", "mahalanobis"):
        raise ValueError("metric must be 'euclidean' or 'mahalanobis'")
    if metric == "mahalanobis":
        if C is None:
            raise ValueError("the Mahalanobis metric needs the "
                             "covariance C")
        Ci = _inv(_mat(C))

        def dist(r):
            d = [r[i] - q[i] for i in range(p)]
            return sqrt(max(0.0, fsum(
                d[i] * fsum(Ci[i][j] * d[j] for j in range(p))
                for i in range(p))))
    else:
        def dist(r):
            return sqrt(fsum((r[i] - q[i]) ** 2 for i in range(p)))

    d = [(dist(Xs[i]), ys[i], i) for i in range(len(Xs))]
    d.sort(key=lambda t: t[0])
    near = d[:kk]
    votes = {}
    for dd, lab, _ in near:
        votes[lab] = votes.get(lab, 0) + 1
    top = max(votes.values())
    tied = [lab for lab, v in votes.items() if v == top]
    if len(tied) == 1:
        winner = tied[0]
    else:
        best, winner = None, None
        for lab in tied:
            s = fsum(dd for dd, l2, _ in near if l2 == lab)
            if best is None or s < best:
                best, winner = s, lab
    return RichResult(payload={
        "assigned": winner, "votes": votes, "k": kk, "metric": metric,
        "neighbours": [{"index": i, "label": lab, "distance": dd}
                       for dd, lab, i in near],
        "tie": len(tied) > 1, "tied_classes": tied,
        "nearest_distance": near[0][0], "nearest_label": near[0][1],
        "single_neighbour_may_be_an_outlier": kk == 1,
        "method": "Rangayyan (2024) eq. (10.29) and Section 10.4.4"})


rangayyan_knn_classifier = knn  # pre-policy spelling


# -- rgksv: K-SVD dictionary learning algorithm.
def ksvdfit(Y, natoms, sparsity, maxiter=15, tol=1e-10, seed=1):
    """Learn an overcomplete dictionary by K-SVD.

    Why: a dictionary assembled from analytic functions represents whatever
    those functions happen to match.  K-SVD instead alternates sparse coding
    with an atom-by-atom update that re-fits each atom to exactly the signals
    that currently use it, so the learned atoms take the shape of the recurring
    structures in the data.

    Alternates: (i) sparse-code every training signal against the current
    dictionary by orthogonal matching pursuit, subject to ||x_i||_0 <= T;
    (ii) for each atom k, form the error matrix restricted to the signals that
    use it and replace the atom and its coefficients by the leading singular
    triplet of that matrix, computed here by power iteration.

    Not from Rangayyan: *Biomedical Signal Analysis*, 3rd ed. Section 9.5
    presents EMD-based dictionary learning (Algorithm 9.1) and cites only the
    label-consistent K-SVD variant, as reference [93] of Chapter 9.  The
    primary source for the algorithm implemented here is Aharon, Elad and
    Bruckstein, "K-SVD: An algorithm for designing overcomplete dictionaries
    for sparse representation", IEEE Transactions on Signal Processing
    54(11):4311-4322, 2006.

    Parameters
    ----------
    Y : sequence of sequences
        Training signals, one signal per row.
    natoms : int
        Dictionary size.
    sparsity : int
        Maximum nonzero coefficients per signal, T.
    maxiter : int
        Alternation rounds.
    tol : float
        Stop when the Frobenius residual stops improving by this much.
    seed : int
        Seed for the deterministic initial dictionary.

    Returns
    -------
    RichResult
        Keys ``dictionary``, ``coefficients``, ``error``, ``iterations``,
        ``method``.
    """
    S = _bxmat(Y, "Y")
    m, n = len(S), len(S[0])
    natoms = int(natoms)
    sparsity = int(sparsity)
    if natoms < 1:
        raise ValueError("natoms must be a positive integer")
    if not (1 <= sparsity <= natoms):
        raise ValueError("sparsity must satisfy 1 <= sparsity <= natoms")
    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    u = _bxrng(seed)
    D = []
    for k in range(natoms):
        a = list(S[k % m]) if k < m else [u() - 0.5 for _ in range(n)]
        a = [t + 1e-3 * (u() - 0.5) for t in a]
        nr = _bxnrm(a)
        D.append([t / nr for t in a] if nr > 1e-12
                 else [1.0 if i == k % n else 0.0 for i in range(n)])

    prev, err, it = None, float("nan"), 0
    Xc = [[0.0] * natoms for _ in range(m)]
    for it in range(1, maxiter + 1):
        Xc = [_bxomp(S[i], D, sparsity, 1e-12)[0] for i in range(m)]
        for k in range(natoms):
            users = [i for i in range(m) if Xc[i][k] != 0.0]
            if not users:
                worst = max(range(m), key=lambda i: _bxnrm(
                    [S[i][t] - fsum(Xc[i][j] * D[j][t] for j in range(natoms))
                     for t in range(n)]))
                nr = _bxnrm(S[worst])
                if nr > 1e-12:
                    D[k] = [t / nr for t in S[worst]]
                continue
            E = []
            for i in users:
                E.append([S[i][t] - fsum(Xc[i][j] * D[j][t]
                                         for j in range(natoms) if j != k)
                          for t in range(n)])
            v = list(D[k])
            for _ in range(30):
                w = [_bxdot(row, v) for row in E]
                nv = [fsum(w[r] * E[r][t] for r in range(len(E))) for t in range(n)]
                nrm = _bxnrm(nv)
                if nrm <= 1e-14:
                    break
                v = [t / nrm for t in nv]
            D[k] = v
            for r, i in enumerate(users):
                Xc[i][k] = _bxdot(E[r], v)
        err = sqrt(fsum((S[i][t] - fsum(Xc[i][j] * D[j][t] for j in range(natoms))) ** 2
                        for i in range(m) for t in range(n)))
        if prev is not None and abs(prev - err) <= tol * max(1.0, prev):
            break
        prev = err

    return RichResult(payload={
        "dictionary": D,
        "coefficients": Xc,
        "error": err,
        "iterations": it,
        "method": "K-SVD dictionary learning with OMP sparse coding; Aharon, "
                  "Elad and Bruckstein, IEEE Trans. Signal Processing "
                  "54(11):4311-4322, 2006 (not the EMD-based scheme of "
                  "Rangayyan Section 9.5)",
    })


rangayyan_ksvd = ksvdfit  # pre-policy spelling


# -- rgldsp: Sparse coding given fixed dictionary (OMP/LASSO).
def dictcode(Y, D, sparsity, tol=1e-12):
    """Sparse-code a set of signals against a fixed dictionary.

    Why: dictionary learning and dictionary use are separate steps.  Once a
    trained dictionary exists it can be used to discover the patterns of
    interest in other signals, and the sparse coefficient vectors -- not the
    signals -- become the feature vectors handed to a classifier.  Rangayyan,
    *Biomedical Signal Analysis*, 3rd ed., Section 9.5 describes this second
    stage and its greedy-approximation character; Section 9.8 uses the
    resulting projection coefficients and reconstruction error as features.

    Each signal is coded by orthogonal matching pursuit: at every step take the
    atom of largest normalised correlation with the residue, then re-solve for
    all selected coefficients by least squares so the residue stays orthogonal
    to the whole active set.  The greedy selection is that of Section 9.5; the
    least-squares reprojection is from Pati, Rezaiifar and Krishnaprasad,
    "Orthogonal matching pursuit", Proceedings of the 27th Asilomar Conference
    on Signals, Systems and Computers, pp. 40-44, 1993, which Rangayyan does
    not cover.

    Parameters
    ----------
    Y : sequence of sequences
        Signals to code, one per row.
    D : sequence of sequences
        Dictionary atoms, one per row, each the length of a signal.
    sparsity : int
        Maximum nonzero coefficients per signal, T.
    tol : float
        Residual norm at which coding of a signal stops early.

    Returns
    -------
    RichResult
        Keys ``coefficients``, ``support``, ``reconstruction``, ``residual``,
        ``error``, ``method``.
    """
    S = _bxmat(Y, "Y")
    A = _bxmat(D, "D")
    n = len(S[0])
    if any(len(a) != n for a in A):
        raise ValueError("dictionary atoms must have the same length as the signals")
    sparsity = int(sparsity)
    if not (1 <= sparsity <= len(A)):
        raise ValueError("sparsity must satisfy 1 <= sparsity <= number of atoms")

    coefs, sups, recs, res = [], [], [], []
    for row in S:
        c, sup, r = _bxomp(row, A, sparsity, float(tol))
        rec = [fsum(c[j] * A[j][i] for j in range(len(A))) for i in range(n)]
        coefs.append(c)
        sups.append(sup)
        recs.append(rec)
        res.append(r)
    err = sqrt(fsum(t * t for r in res for t in r))
    return RichResult(payload={
        "coefficients": coefs,
        "support": sups,
        "reconstruction": recs,
        "residual": res,
        "error": err,
        "method": "sparse coding of signals in a fixed dictionary by orthogonal "
                  "matching pursuit; greedy framework of Rangayyan Biomedical "
                  "Signal Analysis 3rd ed. Section 9.5, orthogonalised per "
                  "Pati, Rezaiifar and Krishnaprasad, Asilomar 1993",
    })


rangayyan_dictionary_sparse = dictcode  # pre-policy spelling


# -- rglindf: Linear discriminant function for pattern classification.
def lindisc(x, weights, w0=None):
    """Linear discriminant and decision functions, Section 10.4.1.

        d_i(x) = w_i^T x + w_i0

    and x is assigned to the class with the LARGEST d_i.  With M classes
    the decision surface between classes i and j is where
    d_i(x) = d_j(x), a hyperplane, so a linear machine carves the feature
    space into convex regions -- which is exactly why it cannot separate
    classes whose regions are not convex, however many features are
    added.

    ``weights`` is one weight vector per class; ``w0`` the matching
    offsets, defaulting to zero.
    """
    xs = aslist(x)
    W = _mat(weights)
    m = len(W)
    if m < 2:
        raise ValueError("need at least two classes")
    if any(len(r) != len(xs) for r in W):
        raise ValueError("every weight vector must match the length of x")
    b = [0.0] * m if w0 is None else aslist(w0)
    if len(b) != m:
        raise ValueError("give one offset per class")
    d = [fsum(W[i][j] * xs[j] for j in range(len(xs))) + b[i]
         for i in range(m)]
    best = max(range(m), key=lambda i: d[i])
    srt = sorted(d, reverse=True)
    return RichResult(payload={
        "d": d, "assigned": best, "margin": srt[0] - srt[1],
        "n_classes": m, "regions_are_convex": True,
        "decision_surfaces_are_hyperplanes": True,
        "method": "Rangayyan (2024) Section 10.4.1 (discriminant and "
                  "decision functions)"})


rangayyan_linear_discrim = lindisc  # pre-policy spelling


# -- rglindsep: Linear discriminant function with optimal separability.
def lindsep(X, y):
    """Linear discriminant with optimal separability, Section 10.4.2.

    Fits the Fisher direction w = S_W^-1 (m1 - m2) and reports where to
    cut the projection.  Two thresholds are offered: the midpoint of the
    projected class means, and the point that actually minimizes the
    training error, which differ whenever the classes have unequal
    spread or unequal size.  The midpoint is only optimal for equal
    priors AND equal variances, so the fitted cut is reported as the
    default and the midpoint alongside for comparison.

    The training error is resubstitution error -- the same data that
    chose the cut -- so it is optimistic by construction.  Section 10.10.3
    is the book's warning on this; use ``kfoldcv`` or ``loocv`` for an
    honest figure.
    """
    f = fishlda(X, y)
    w = f["w"]
    order = f["classes"]
    proj = f["projected"]
    a, b = proj[order[0]], proj[order[1]]
    ma, mb = f["projected_means"]
    mid = 0.5 * (ma + mb)
    hi_first = ma > mb
    cand = sorted(set(a + b))
    best_t, best_err = mid, None
    for i in range(len(cand) + 1):
        lo = cand[0] - 1.0 if i == 0 else cand[i - 1]
        hi = cand[-1] + 1.0 if i == len(cand) else cand[i]
        t = 0.5 * (lo + hi)
        if hi_first:
            err = sum(1 for v in a if v <= t) + sum(1 for v in b if v > t)
        else:
            err = sum(1 for v in a if v > t) + sum(1 for v in b if v <= t)
        if best_err is None or err < best_err:
            best_err, best_t = err, t
    n = len(a) + len(b)
    if hi_first:
        mid_err = sum(1 for v in a if v <= mid) + sum(1 for v in b
                                                     if v > mid)
    else:
        mid_err = sum(1 for v in a if v > mid) + sum(1 for v in b
                                                    if v <= mid)
    return RichResult(payload={
        "w": w, "threshold": best_t, "midpoint_threshold": mid,
        "classes": order, "first_class_is_above": hi_first,
        "training_errors": best_err, "midpoint_errors": mid_err,
        "training_accuracy": 1.0 - best_err / n, "n": n,
        "projected": proj,
        "midpoint_optimal_only_for_equal_priors_and_spread": True,
        "resubstitution_error_is_optimistic": True,
        "method": "Rangayyan (2024) Sections 10.4.2 and 10.10.3"})


rangayyan_lin_discr_sep = lindsep  # pre-policy spelling


# -- rgloo: Leave-one-out cross-validation (LOO-CV).
def loocv(X, y, classifier=None):
    """Leave-one-out cross-validation, Section 10.10.3.

        LOO = (1/N) sum_i I( f_{-i}(x_i) != y_i )

    K-fold with K = N.  It uses the most training data of any split, so
    its estimate is nearly unbiased, and it is deterministic -- there is
    only one way to leave one out, so unlike 5-fold it gives the same
    answer every time.

    The cost is N fits, and a high variance: each of the N training sets
    differs from the others by a single sample, so the errors are heavily
    correlated and the estimate moves a lot from dataset to dataset.  For
    a small biomedical study, which is the case the book is concerned
    with, that trade is usually worth taking.
    """
    Xs = _mat(X)
    ys = list(y)
    n = len(Xs)
    if n != len(ys):
        raise ValueError("X and y must have the same number of rows")
    if n < 3:
        raise ValueError("need at least three samples")
    if classifier is None:
        def classifier(Xt, yt, q):
            return knn(Xt, yt, q, k=1)["assigned"]
    errors, wrong = 0, []
    for i in range(n):
        Xt = [Xs[j] for j in range(n) if j != i]
        yt = [ys[j] for j in range(n) if j != i]
        if len(set(yt)) < 2:
            raise ValueError("removing sample %d leaves one class; the "
                             "classifier cannot be trained" % i)
        if classifier(Xt, yt, Xs[i]) != ys[i]:
            errors += 1
            wrong.append(i)
    return RichResult(payload={
        "error_rate": errors / n, "accuracy": 1.0 - errors / n,
        "errors": errors, "misclassified": wrong, "n": n, "n_fits": n,
        "deterministic": True, "nearly_unbiased": True,
        "high_variance": True,
        "method": "Rangayyan (2024) Section 10.10.3 (leave-one-out)"})


rangayyan_loo_cv = loocv  # pre-policy spelling


# -- rglr: Logistic regression for binary classification.
def logreg(X, y, maxiter=100, tol=1e-8, ridge=1e-8):
    """Logistic regression, Section 10.7.

        P(y = 1 | x) = 1 / (1 + exp(-(w^T x + b)))

    fitted by Newton-Raphson on the log-likelihood, which is the
    iteratively reweighted least squares of the standard texts.  Unlike
    the Bayes classifier this models the POSTERIOR directly and assumes
    nothing about the shape of p(x|C), which is why it survives features
    that are plainly not Gaussian.

    Perfectly separable classes have no finite maximum -- the likelihood
    keeps rising as the weights grow -- so a small ridge is added and
    ``separable`` is reported when the fit runs to the iteration limit
    with an exploding norm.  Without that, the coefficients are
    meaningless numbers that merely record where the optimizer stopped.
    """
    Xs = _mat(X)
    ys = [float(v) for v in y]
    if len(Xs) != len(ys):
        raise ValueError("X and y must have the same number of rows")
    if any(v not in (0.0, 1.0) for v in ys):
        raise ValueError("logistic regression needs 0/1 labels")
    if len(set(ys)) < 2:
        raise ValueError("both classes must be present")
    n = len(Xs)
    p = len(Xs[0]) + 1
    A = [[1.0] + list(r) for r in Xs]
    w = [0.0] * p
    lam = float(ridge)
    it, sep = 0, False
    for it in range(1, int(maxiter) + 1):
        eta = [fsum(A[i][j] * w[j] for j in range(p)) for i in range(n)]
        mu = [1.0 / (1.0 + exp(-min(500.0, max(-500.0, v)))) for v in eta]
        g = [fsum(A[i][j] * (ys[i] - mu[i]) for i in range(n))
             - lam * w[j] for j in range(p)]
        H = [[0.0] * p for _ in range(p)]
        for i in range(n):
            wt = mu[i] * (1.0 - mu[i])
            for a in range(p):
                for b in range(p):
                    H[a][b] += wt * A[i][a] * A[i][b]
        for a in range(p):
            H[a][a] += lam
        try:
            step = _solve_lin(H, g)
        except ValueError:
            break
        w = [w[j] + step[j] for j in range(p)]
        if max(abs(v) for v in step) < tol:
            break
    norm = sqrt(fsum(v * v for v in w))
    if it >= int(maxiter) and norm > 50.0:
        sep = True
    eta = [fsum(A[i][j] * w[j] for j in range(p)) for i in range(n)]
    mu = [1.0 / (1.0 + exp(-min(500.0, max(-500.0, v)))) for v in eta]
    ll = fsum(ys[i] * log(max(mu[i], 1e-300))
              + (1 - ys[i]) * log(max(1 - mu[i], 1e-300))
              for i in range(n))
    pred = [1 if v >= 0.5 else 0 for v in mu]
    acc = sum(1 for i in range(n) if pred[i] == ys[i]) / n
    return RichResult(payload={
        "intercept": w[0], "coefficients": w[1:], "w": w,
        "fitted": mu, "predicted": pred, "loglik": ll,
        "iterations": it, "converged": it < int(maxiter),
        "separable": sep, "ridge": lam,
        "training_accuracy": acc, "n": n,
        "models_the_posterior_directly": True,
        "no_gaussian_assumption": True,
        "method": "Rangayyan (2024) Section 10.7 (logistic regression)"})


rangayyan_logistic_regression = logreg  # pre-policy spelling


# -- rglstm: LSTM recurrent network for biomedical time-series classification.
def lstm(sequences, labels=None, hidden=8, ridge=1e-6, seed=1, weights=None):
    """Long short-term memory recurrence over a biomedical time series.

    Why: a plain recurrent unit forgets across long stretches, but the clinical
    events that matter in a biomedical recording -- an apnoea, a seizure onset,
    a run of ectopic beats -- are separated by long, uninformative intervals.
    The gated cell keeps a state that survives those intervals.

    Not from Rangayyan: *Biomedical Signal Analysis*, 3rd ed., Section 10.8.2
    discusses deep learning in prose and names convolutional networks, but
    gives no recurrent or gated-cell equations.  The primary source is
    Hochreiter and Schmidhuber, "Long short-term memory", Neural Computation
    9(8):1735-1780, 1997, with the forget gate of Gers, Schmidhuber and
    Cummins, Neural Computation 12(10):2451-2471, 2000:

        i = sigmoid(W_i [h, x] + b_i),   f = sigmoid(W_f [h, x] + b_f),
        o = sigmoid(W_o [h, x] + b_o),   g = tanh(W_g [h, x] + b_g),
        c = f * c + i * g,               h = o * tanh(c).

    The recurrent weights are fixed (supplied, or drawn once from the seeded
    generator) and only the linear readout on the final hidden state is fitted,
    by ridge least squares.  That keeps the whole routine deterministic and
    closed-form; it is a reservoir-style readout, not back-propagation through
    time, and the docstring says so rather than implying a trained network.

    Parameters
    ----------
    sequences : sequence
        One sequence per sample; each is a list of time steps, and each step is
        a scalar or a list of input features.
    labels : sequence, optional
        Class code per sequence.  Without labels only the states are returned.
    hidden : int
        Cell width.
    ridge : float
        Ridge term of the readout least squares.
    seed : int
        Seed for the fixed recurrent weights.
    weights : dict, optional
        Supplied gate weights, keys ``i``, ``f``, ``o``, ``g``, each an
        ``hidden`` x (hidden + inputs) matrix, plus ``bias`` with one row of
        length ``hidden`` per gate in that order.

    Returns
    -------
    RichResult
        Keys ``hidden``, ``cell``, ``predictions``, ``accuracy``,
        ``readout``, ``classes``, ``method``.
    """
    try:
        seqs = [list(s) for s in sequences]
    except TypeError:
        raise ValueError("sequences must be a sequence of sequences")
    if not seqs:
        raise ValueError("sequences must be non-empty")
    steps = []
    for s in seqs:
        if not s:
            raise ValueError("every sequence must have at least one time step")
        steps.append([aslist(t) for t in s])
    d = len(steps[0][0])
    for s in steps:
        if any(len(t) != d for t in s):
            raise ValueError("all time steps must have the same number of inputs")
    H = int(hidden)
    if H < 1:
        raise ValueError("hidden must be a positive integer")

    if weights is None:
        u = _bxrng(seed)
        sc = 1.0 / sqrt(H + d)
        W = {k: [[sc * (2.0 * u() - 1.0) for _ in range(H + d)] for _ in range(H)]
             for k in ("i", "f", "o", "g")}
        B = {"i": [0.0] * H, "f": [1.0] * H, "o": [0.0] * H, "g": [0.0] * H}
    else:
        if not isinstance(weights, dict):
            raise ValueError("weights must be a dict of gate matrices")
        W = {}
        for k in ("i", "f", "o", "g"):
            if k not in weights:
                raise ValueError("weights is missing gate %r" % k)
            M = _bxmat(weights[k], "weights[%r]" % k)
            if len(M) != H or len(M[0]) != H + d:
                raise ValueError("weights[%r] must be %d x %d" % (k, H, H + d))
            W[k] = M
        bb = _bxmat(weights.get("bias", [[0.0] * H] * 4), "weights['bias']")
        if len(bb) != 4 or any(len(r) != H for r in bb):
            raise ValueError("weights['bias'] must be 4 rows of length %d" % H)
        B = {"i": bb[0], "f": bb[1], "o": bb[2], "g": bb[3]}

    def sig(b):
        if b < -700.0:
            return 0.0
        if b > 700.0:
            return 1.0
        return 1.0 / (1.0 + exp(-b))

    hs, cs = [], []
    for s in steps:
        h = [0.0] * H
        c = [0.0] * H
        for x in s:
            z = h + x
            i = [sig(_bxdot(W["i"][k], z) + B["i"][k]) for k in range(H)]
            f = [sig(_bxdot(W["f"][k], z) + B["f"][k]) for k in range(H)]
            o = [sig(_bxdot(W["o"][k], z) + B["o"][k]) for k in range(H)]
            g = [tanh(_bxdot(W["g"][k], z) + B["g"][k]) for k in range(H)]
            c = [f[k] * c[k] + i[k] * g[k] for k in range(H)]
            h = [o[k] * tanh(c[k]) for k in range(H)]
        hs.append(h)
        cs.append(c)

    pred, acc, read, classes = None, float("nan"), None, None
    if labels is not None:
        y = [int(t) for t in _bxvec(labels, "labels")]
        if len(y) != len(seqs):
            raise ValueError("labels must have one entry per sequence")
        classes = sorted(set(y))
        if len(classes) < 2:
            raise ValueError("labels must contain at least two classes")
        A = [row + [1.0] for row in hs]
        read = []
        for c in classes:
            t = [1.0 if y[i] == c else 0.0 for i in range(len(y))]
            read.append(_bxlstsq(A, t, float(ridge)))
        pred = []
        for i in range(len(y)):
            sc = [_bxdot(w, A[i]) for w in read]
            pred.append(classes[max(range(len(classes)), key=lambda k: sc[k])])
        acc = fsum(1.0 for a, b in zip(y, pred) if a == b) / len(y)

    return RichResult(payload={
        "hidden": hs,
        "cell": cs,
        "predictions": pred,
        "accuracy": acc,
        "readout": read,
        "classes": classes,
        "method": "LSTM recurrence with a ridge least-squares readout on the "
                  "final hidden state; Hochreiter and Schmidhuber, Neural "
                  "Computation 9(8):1735-1780, 1997, with the forget gate of "
                  "Gers, Schmidhuber and Cummins, Neural Computation "
                  "12(10):2451-2471, 2000 (not covered by Rangayyan)",
    })


rangayyan_lstm_signal = lstm  # pre-policy spelling


# -- rgmahd: Mahalanobis distance from sample to class.
def mahal(x, mu, C):
    """Mahalanobis distance, Section 10.4.3.

        D^2 = (x - mu)^T C^-1 (x - mu)

    The distance in units of the data's own scatter.  Where the Euclidean
    distance treats every direction alike, this one shrinks the
    directions the class already varies in, so a point far away along an
    axis of natural variation is NEAR in this metric and a point close by
    across the grain is far.  That is why it is the right distance for
    classifying against a class with correlated features, and why
    substituting Euclidean distance quietly favours whichever feature
    happens to have the largest units.

    The squared distance is the primary result; the book's distance
    functions use the square, and taking a root is only meaningful
    because the form is positive definite.
    """
    xs, m = aslist(x), aslist(mu)
    S = _mat(C)
    p = len(xs)
    if len(m) != p:
        raise ValueError("x and mu must have the same length")
    if len(S) != p or any(len(r) != p for r in S):
        raise ValueError("the covariance must be %d x %d" % (p, p))
    Si = _inv(S)
    d = [xs[i] - m[i] for i in range(p)]
    d2 = fsum(d[i] * fsum(Si[i][j] * d[j] for j in range(p))
              for i in range(p))
    eucl = sqrt(fsum(v * v for v in d))
    return RichResult(payload={
        "d2": d2, "distance": sqrt(d2) if d2 >= 0 else float("nan"),
        "squared": d2, "euclidean": eucl,
        "differs_from_euclidean": abs(sqrt(max(d2, 0.0)) - eucl) > 1e-12,
        "scale_free": True,
        "method": "Rangayyan (2024) Section 10.4.3 (distance functions)"})


rangayyan_mahalanobis = mahal  # pre-policy spelling


# -- rgmcn: McNemar's test for comparing two classifiers.
def mcnemar(table, correct=None):
    """McNemar's test of SYMMETRY, Section 10.9.2.

    The book states the test on a general contingency table comparing two
    methods -- its worked example, Table 10.4, is 3x3 with categories
    normal, indeterminate and abnormal -- so this accepts any k x k
    table, not only 2x2.  For k = 2 the statistic is McNemar's

        chi2 = (|b - c| - 1)^2 / (b + c),   df = 1

    with Yates' continuity correction, and for k > 2 its generalization,
    Bowker's test of symmetry,

        chi2 = sum_{i<j} (n_ij - n_ji)^2 / (n_ij + n_ji),
        df = k(k-1)/2.

    Only the OFF-DIAGONAL disagreements enter.  The diagonal -- the cases
    both methods called the same way, which is usually most of them --
    contributes nothing, which is the whole point: the question is
    whether the disagreements are one-sided, not how often the methods
    agree.

    Pairs where both cells are zero contribute nothing and are excluded
    from the degrees of freedom, since they carry no information.
    """
    t = _mat(table)
    k = len(t)
    if k < 2 or any(len(r) != k for r in t):
        raise ValueError("the table must be square and at least 2x2")
    if any(v < 0 for r in t for v in r):
        raise ValueError("counts cannot be negative")
    yates = (k == 2) if correct is None else bool(correct)
    stat, df, pairs = 0.0, 0, []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = t[i][j], t[j][i]
            if a + b <= 0:
                continue
            d = abs(a - b)
            if yates:
                d = max(0.0, d - 1.0)
            stat += d * d / (a + b)
            df += 1
            pairs.append({"i": i, "j": j, "n_ij": a, "n_ji": b})
    if df == 0:
        raise ValueError("the table is symmetric with no off-diagonal "
                         "counts; the test is undefined")
    p = _chisq_sf(stat, df)
    n = fsum(v for r in t for v in r)
    diag = fsum(t[i][i] for i in range(k))
    return RichResult(payload={
        "statistic": stat, "df": df, "p_value": p,
        "pairs": pairs, "n": n, "n_agree": diag,
        "continuity_correction": yates,
        "is_bowker": k > 2, "k": k,
        "diagonal_contributes_nothing": True,
        "method": "Rangayyan (2024) Section 10.9.2 (McNemar's test of "
                  "symmetry; Bowker's generalization for k > 2)"})


rangayyan_mcnemar_test = mcnemar  # pre-policy spelling


# -- rgmp: Matching pursuit greedy decomposition into dictionary atoms.
def mpursuit(x, dictionary=None, natoms=20, tol=1e-10, decaystop=None):
    """Decompose a signal by matching pursuit into time-frequency atoms.

    Why: a fixed transform imposes one tiling of the time-frequency plane on
    every signal.  Matching pursuit instead picks, at each step, whichever
    dictionary atom currently matches the signal best, so the representation
    adapts to the signal.  What it selects are the coherent structures present
    in the signal, and what is left over may be taken as random noise, which is
    why the truncated expansion works as an adaptive filter.  Rangayyan,
    *Biomedical Signal Analysis*, 3rd ed., Section 9.3.

    First projection, eq. (9.4):
        x(t) = <x, g_0> g_0(t) + R_1 x(t),
    then after M iterations, eq. (9.5), with R_0 x(t) = x(t):
        x(t) = sum_{n<M} <R_n x, g_n> g_n(t) + R_M x(t).
    The reconstruction from the M selected structures is eq. (9.7), and the
    decay parameter of eq. (9.6),
        lambda(m) = sqrt(1 - ||R_m x||^2 / ||R_(m-1) x||^2),
    is returned so the caller can stop once it no longer falls appreciably.

    The default dictionary is the Gabor dictionary of eqs. (9.2) and (9.3),
    the Gaussian window g(t) = 2^(1/4) exp(-pi t^2) scaled, translated and
    modulated on a dyadic grid.

    Parameters
    ----------
    x : sequence
        Signal to decompose.
    dictionary : sequence of sequences, optional
        Atoms, one per row, each the length of x.  Gabor atoms by default.
    natoms : int
        Maximum number M of atoms to select.
    tol : float
        Stop when the residue energy falls below this.
    decaystop : float, optional
        Stop once the decay parameter of eq. (9.6) falls below this.

    Returns
    -------
    RichResult
        Keys ``coefficients``, ``atoms``, ``indices``, ``residual``,
        ``reconstruction``, ``decay``, ``energyratio``, ``parameters``,
        ``method``.
    """
    x = _bxvec(x, "x")
    n = len(x)
    natoms = int(natoms)
    if natoms < 1:
        raise ValueError("natoms must be a positive integer")
    if dictionary is None:
        D, params = _bxgabor(n, max(64, 8 * natoms))
    else:
        D = _bxmat(dictionary, "dictionary")
        if any(len(a) != n for a in D):
            raise ValueError("every atom must have the same length as x")
        params = [None] * len(D)
        nd = []
        for a in D:
            nr = _bxnrm(a)
            if nr <= 1e-12:
                raise ValueError("dictionary atoms must have nonzero norm")
            nd.append([t / nr for t in a])
        D = nd
    if not D:
        raise ValueError("the dictionary is empty")

    r = list(x)
    e_prev = fsum(t * t for t in r)
    if e_prev <= 0.0:
        raise ValueError("x has zero energy")
    e0 = e_prev
    coef, idx, decay = [], [], []
    for _ in range(min(natoms, len(D))):
        best, bv = -1, -1.0
        for j in range(len(D)):
            if j in idx:
                continue
            v = abs(_bxdot(D[j], r))
            if v > bv:
                best, bv = j, v
        if best < 0:
            break
        a = _bxdot(D[best], r)
        r = [r[i] - a * D[best][i] for i in range(n)]
        coef.append(a)
        idx.append(best)
        e_now = fsum(t * t for t in r)
        lam = sqrt(max(0.0, 1.0 - e_now / e_prev)) if e_prev > 0.0 else 0.0
        decay.append(lam)
        e_prev = e_now
        if e_now <= tol:
            break
        if decaystop is not None and lam < float(decaystop):
            break

    rec = [fsum(coef[k] * D[idx[k]][i] for k in range(len(idx))) for i in range(n)]
    return RichResult(payload={
        "coefficients": coef,
        "atoms": [D[j] for j in idx],
        "indices": idx,
        "residual": r,
        "reconstruction": rec,
        "decay": decay,
        "energyratio": 1.0 - e_prev / e0,
        "parameters": [params[j] for j in idx],
        "method": "matching-pursuit decomposition into time-frequency atoms, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 9.3, "
                  "eqs. (9.1)-(9.7) with the Gabor dictionary of eqs. (9.2)-(9.3)",
    })


rangayyan_matching_pursuit = mpursuit  # pre-policy spelling


# -- rgneural: Neural decoding for prosthesis control from spike trains.
def bmidec(y, C, a=None, procnoise=1e-4, obsnoise=1e-2, p0=1e-2):
    """Decode intended movement from neural observations with a Kalman filter.

    Why: a brain-machine interface has no ground truth about the user's
    intended hand kinematics -- that is precisely what is missing in motor
    impairment -- so the decoder must estimate a hidden state from noisy
    multichannel observations, recursively and in real time.  The Kalman filter
    does exactly that and is preferred to RLS for motion control because of its
    dynamic tracking and real-time behaviour.  Rangayyan, *Biomedical Signal
    Analysis*, 3rd ed., Section 8.18, using the filter of Section 8.7.

    Process and observation models, eqs. (8.60) and (8.63):
        x(n+1) = a(n+1, n) x(n) + eta_d(n),
        y(n)   = C(n) x(n) + eta_o(n).
    The recursion is the book's five steps, eqs. (8.95) to (8.99), with the
    stated initial conditions xtilde(1|Y_0) = 0 and phi_ep(1, 0) = D_0, a
    diagonal matrix with values of the order of 1e-2.

    Parameters
    ----------
    y : sequence of sequences
        Observations, one time step per row, each of length K (channels).
    C : sequence of sequences
        Observation matrix, K x L, mapping the state to the observation.
    a : sequence of sequences, optional
        State transition matrix, L x L; the identity by default.
    procnoise : float or sequence of sequences
        Driving-noise covariance phi_etad, scalar (times the identity) or full.
    obsnoise : float or sequence of sequences
        Observation-noise covariance phi_etao, scalar or full.
    p0 : float
        Diagonal value of D_0.

    Returns
    -------
    RichResult
        Keys ``states``, ``innovations``, ``gain``, ``predicted``, ``method``.
    """
    Y = _bxmat(y, "y")
    Cm = _bxmat(C, "C")
    K, L = len(Cm), len(Cm[0])
    if len(Y[0]) != K:
        raise ValueError("each observation row must have %d entries" % K)
    if a is None:
        A = [[1.0 if i == j else 0.0 for j in range(L)] for i in range(L)]
    else:
        A = _bxmat(a, "a")
        if len(A) != L or len(A[0]) != L:
            raise ValueError("a must be %d x %d" % (L, L))
    if float(p0) <= 0.0:
        raise ValueError("p0 must be positive")

    def cov(arg, k, name):
        if isinstance(arg, (int, float)):
            if float(arg) <= 0.0:
                raise ValueError(name + " must be positive")
            return [[float(arg) if i == j else 0.0 for j in range(k)]
                    for i in range(k)]
        M = _bxmat(arg, name)
        if len(M) != k or len(M[0]) != k:
            raise ValueError(name + " must be %d x %d" % (k, k))
        return M

    Qd = cov(procnoise, L, "procnoise")
    Qo = cov(obsnoise, K, "obsnoise")

    xh = [0.0] * L
    P = [[float(p0) if i == j else 0.0 for j in range(L)] for i in range(L)]
    Ct = _bxtr(Cm)
    states, innov, gains, preds = [], [], [], []
    for t in range(len(Y)):
        PCt = _bxmm(P, Ct)
        Sm = _bxmm(Cm, PCt)
        for i in range(K):
            for j in range(K):
                Sm[i][j] += Qo[i][j]
        Kg = []
        APCt = _bxmm(A, PCt)
        for col in range(K):
            e = [1.0 if r == col else 0.0 for r in range(K)]
            Sinv_col = _bxsolve(Sm, e)
            Kg.append(Sinv_col)
        Kmat = [[fsum(APCt[i][k] * Kg[j][k] for k in range(K)) for j in range(K)]
                for i in range(L)]
        pred = _bxmv(Cm, xh)
        z = [Y[t][i] - pred[i] for i in range(K)]
        preds.append(pred)
        innov.append(z)
        states.append(list(xh))
        gains.append([list(r) for r in Kmat])
        xh = [fsum(A[i][j] * xh[j] for j in range(L))
              + fsum(Kmat[i][j] * z[j] for j in range(K)) for i in range(L)]
        KC = _bxmm(Kmat, Cm)
        Pf = [[P[i][j] - fsum(KC[i][k] * P[k][j] for k in range(L))
               for j in range(L)] for i in range(L)]
        P = _bxmm(_bxmm(A, Pf), _bxtr(A))
        for i in range(L):
            for j in range(L):
                P[i][j] += Qd[i][j]

    return RichResult(payload={
        "states": states,
        "innovations": innov,
        "gain": gains,
        "predicted": preds,
        "method": "Kalman-filter neural decoder for prosthesis control, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 8.18 "
                  "with the recursion of Section 8.7, eqs. (8.60), (8.63) and "
                  "(8.95)-(8.99)",
    })


rangayyan_neural_decode = bmidec  # pre-policy spelling


# -- rgnmf: Nonnegative matrix factorization (NMF) with multiplicative update rules.
def nmfmu(V, r, maxiter=200, tol=1e-10, cost="ls", seed=1):
    """Nonnegative matrix factorisation by multiplicative updates.

    Why: PCA and ICA are free to use negative coefficients, so their components
    cancel one another and lose any parts-based reading.  Constraining both
    factors to be nonnegative makes the columns of W act as basis vectors and
    the rows of H as the weights or activations that switch them on, which is
    the interpretation the signal analyst wants from a time-frequency matrix.
    Rangayyan, *Biomedical Signal Analysis*, 3rd ed., Section 9.7.3.

    Finds V approximately equal to W H, eq. (9.46), with W of size M x r and
    H of size r x N, r < min(M, N).  Squared-error cost gives the updates of
    eqs. (9.49) and (9.50):

        H <- H .* (W' V) ./ (W' W H),   W <- W .* (V H') ./ (W H H'),

    with the elementwise product and quotient of eqs. (9.51) and (9.52).
    The divergence cost of eq. (9.53) gives eqs. (9.54) and (9.55) instead;
    the book warns that the divergence form is undefined where an element of
    V or W H is zero, so it should be used with caution.

    Parameters
    ----------
    V : sequence of sequences
        Nonnegative M x N matrix.
    r : int
        Factorisation rank.
    maxiter : int
        Maximum update iterations.
    tol : float
        Relative change in the residual at which iteration stops.
    cost : {"ls", "kld"}
        Squared error, eqs. (9.49)-(9.50), or divergence, eqs. (9.54)-(9.55).
    seed : int
        Seed for the deterministic initialisation.

    Returns
    -------
    RichResult
        Keys ``W``, ``H``, ``submatrices``, ``error``, ``iterations``,
        ``cost``, ``method``.
    """
    M = _bxmat(V, "V")
    if cost not in ("ls", "kld"):
        raise ValueError("cost must be 'ls' or 'kld'")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be a positive integer")
    if cost == "kld" and any(t <= 0.0 for row in M for t in row):
        raise ValueError("the divergence cost is undefined where V has a zero "
                         "element; use cost='ls'")
    W, H, err, it = _bxnmfmu(M, r, int(maxiter), float(tol), seed, cost)
    subs = [[[W[i][k] * H[k][j] for j in range(len(H[0]))]
             for i in range(len(W))] for k in range(len(H))]
    return RichResult(payload={
        "W": W,
        "H": H,
        "submatrices": subs,
        "error": err,
        "iterations": it,
        "cost": cost,
        "method": "nonnegative matrix factorisation by multiplicative updates, "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 9.7.3, "
                  "eqs. (9.46), (9.49)-(9.50) and (9.53)-(9.55)",
    })


rangayyan_nmf = nmfmu  # pre-policy spelling


# -- rgnmfch: NMF-based EEG channel selection for BCI.
def nmfchsel(trials, nselect, rank=4, maxiter=200, tol=1e-8, seed=1):
    """Rank EEG channels by the NMF basis-row deviation score.

    Why: multichannel EEG carries redundant and noisy channels; feeding them
    all to a classifier adds complexity and invites overfitting, and because
    channel relevance varies strongly between subjects a fixed montage is the
    wrong answer.  Factorising the channel covariance matrix scores each
    channel by how distinctive its loading pattern is.  Rangayyan, *Biomedical
    Signal Analysis*, 3rd ed., Section 9.12.1.

    Steps: build the N x N channel covariance matrix of eq. (9.94) from the
    N x T trial matrix; factorise it by NMF; min-max normalise each row of the
    basis matrix by eq. (9.95),

        W_j <- (W_j - min W_j) / (max W_j - min W_j);

    then score the row by its RMS deviation from W_ref, the vector with all
    elements 0.5, eq. (9.96).  A row that stays near 0.5 loads uniformly on
    every factor and carries little that is specific; a large deviation marks a
    channel with a distinctive spatial pattern.

    Rank note: with r = 2 the min-max normalisation of eq. (9.95) sends every
    two-element row to {0, 1}, so eq. (9.96) returns exactly 0.5 for every
    channel and the score ranks nothing.  A rank below 3 is therefore rejected.

    Note on eq. (9.96): as printed, the index j appears both as the row index
    of W_j and as the summation index, which cannot be read literally.  This
    implementation takes the per-row RMS deviation, that is the root mean over
    the r factors of (W_jk - 0.5)^2, which is the reading consistent with the
    surrounding text.

    ``bcichsel`` wraps this for the full BCI application and additionally
    applies the channel weights to the selected signals.

    Parameters
    ----------
    trials : sequence of sequences
        One trial as an N x T matrix, N channels by T samples.
    nselect : int
        How many channels to return.
    rank : int
        Factorisation rank r.
    maxiter, tol : int, float
        Update budget and tolerance.
    seed : int
        Seed for the deterministic initialisation.

    Returns
    -------
    RichResult
        Keys ``selected``, ``rmsd``, ``ranking``, ``normalized``, ``W``,
        ``H``, ``covariance``, ``method``.
    """
    X = _bxmat(trials, "trials")
    nch = len(X)
    if nch < 2:
        raise ValueError("need at least two EEG channels")
    nselect = int(nselect)
    if not (1 <= nselect <= nch):
        raise ValueError("nselect must satisfy 1 <= nselect <= number of channels")
    rank = int(rank)
    if rank < 3:
        raise ValueError("rank must be at least 3: with r = 2 the min-max "
                         "normalisation of eq. (9.95) maps every basis row to "
                         "{0, 1}, so the RMS deviation of eq. (9.96) is exactly "
                         "0.5 for every channel and ranks nothing")

    mu, C = _bxcov(_bxtr(X))
    shift = min(min(r) for r in C)
    V = [[t - shift for t in r] for r in C] if shift < 0.0 else [list(r) for r in C]
    W, H, err, _ = _bxnmfmu(V, rank, maxiter, tol, seed, "ls")

    rmsd, Wn = [], []
    for j in range(nch):
        row = W[j]
        lo, hi = min(row), max(row)
        nr = [0.5] * len(row) if hi - lo <= 0 else [(t - lo) / (hi - lo) for t in row]
        Wn.append(nr)
        rmsd.append(sqrt(fsum((t - 0.5) ** 2 for t in nr) / len(nr)))

    ranking = sorted(range(nch), key=lambda i: (-rmsd[i], i))
    return RichResult(payload={
        "selected": sorted(ranking[:nselect]),
        "rmsd": rmsd,
        "ranking": ranking,
        "normalized": Wn,
        "W": W,
        "H": H,
        "covariance": C,
        "error": err,
        "method": "NMF-based EEG channel ranking by normalised basis-row RMS "
                  "deviation, Rangayyan Biomedical Signal Analysis 3rd ed. "
                  "Section 9.12.1, eqs. (9.94)-(9.96)",
    })


rangayyan_nmf_channel_sel = nmfchsel  # pre-policy spelling


# -- rgomp: Orthogonal matching pursuit (OMP) for sparse representation.
def ompfit(x, D, sparsity=None, tol=1e-10):
    """Orthogonal matching pursuit of one signal in a dictionary.

    Why: plain matching pursuit subtracts only the component along the newly
    chosen atom, so with correlated atoms it revisits nearly the same direction
    again and again and converges slowly.  Re-solving all selected coefficients
    by least squares after every pick keeps the residue orthogonal to the whole
    active set, so each atom is chosen at most once and k atoms give the best
    k-term fit on that support.

    Not from Rangayyan: *Biomedical Signal Analysis*, 3rd ed. presents matching
    pursuit in Section 9.3 (eqs. 9.1-9.7) but not its orthogonalised variant.
    The primary source is Pati, Rezaiifar and Krishnaprasad, "Orthogonal
    matching pursuit: recursive function approximation with applications to
    wavelet decomposition", Proceedings of the 27th Asilomar Conference on
    Signals, Systems and Computers, pp. 40-44, 1993.

    Parameters
    ----------
    x : sequence
        Signal to represent.
    D : sequence of sequences
        Dictionary atoms, one per row, each the length of x.
    sparsity : int, optional
        Maximum number of atoms; unbounded by default, in which case ``tol``
        alone stops the loop.
    tol : float
        Residual norm at which the loop stops.

    Returns
    -------
    RichResult
        Keys ``coefficients``, ``support``, ``reconstruction``, ``residual``,
        ``error``, ``energyratio``, ``method``.
    """
    x = _bxvec(x, "x")
    A = _bxmat(D, "D")
    n = len(x)
    if any(len(a) != n for a in A):
        raise ValueError("every dictionary atom must have the same length as x")
    e0 = fsum(t * t for t in x)
    if e0 <= 0.0:
        raise ValueError("x has zero energy")
    c, sup, r = _bxomp(x, A, sparsity, float(tol))
    rec = [fsum(c[j] * A[j][i] for j in range(len(A))) for i in range(n)]
    err = _bxnrm(r)
    return RichResult(payload={
        "coefficients": c,
        "support": sup,
        "reconstruction": rec,
        "residual": r,
        "error": err,
        "energyratio": 1.0 - (err * err) / e0,
        "method": "orthogonal matching pursuit; Pati, Rezaiifar and "
                  "Krishnaprasad, Proc. 27th Asilomar Conf., pp. 40-44, 1993 "
                  "(Rangayyan Section 9.3 covers plain matching pursuit)",
    })


rangayyan_omp = ompfit  # pre-policy spelling


# -- rgpca: PCA for signal mixture separation (eigendecomposition of covariance).
def pcasig(X, ncomp=None):
    """Principal component analysis of a set of correlated signals.

    Why: multichannel biomedical recordings pick up the same sources through
    different paths, so the channels carry redundant, correlated information.
    Rotating onto the eigenvectors of the covariance matrix produces
    uncorrelated components ordered by variance, so most of the power lands in
    a few components and the rest can be dropped with the smallest possible
    mean squared error.  Rangayyan, *Biomedical Signal Analysis*, 3rd ed.,
    Section 9.7.1.

    Truncating to L of K components leaves the error of eq. (9.37); choosing
    the basis vectors as the eigenvectors of the covariance, eq. (9.38), with
    eigenvalues eq. (9.39), makes that error the sum of the discarded
    eigenvalues, eq. (9.40).  Ordering the eigenvalues in decreasing order
    therefore minimises the mean squared error for any chosen L, and the
    transformed components are mutually uncorrelated with the eigenvalues as
    their variances, eq. (9.41).

    Parameters
    ----------
    X : sequence of sequences
        Signals, one channel per row, samples along the row.
    ncomp : int, optional
        Number of components to keep; all of them by default.

    Returns
    -------
    RichResult
        Keys ``components``, ``eigenvalues``, ``eigenvectors``, ``mean``,
        ``varexplained``, ``mse``, ``method``.
    """
    Y = _bxmat(X, "X")
    K, T = len(Y), len(Y[0])
    if T < 2:
        raise ValueError("need at least two samples per channel")
    L = K if ncomp is None else int(ncomp)
    if not (1 <= L <= K):
        raise ValueError("ncomp must satisfy 1 <= ncomp <= number of channels")

    mu = [_bxmean(r) for r in Y]
    Yc = [[Y[i][t] - mu[i] for t in range(T)] for i in range(K)]
    S = [[fsum(Yc[i][t] * Yc[j][t] for t in range(T)) / (T - 1) for j in range(K)]
         for i in range(K)]
    vals, vecs = _bxjacobi(S)
    W = [[vecs[j][k] for j in range(K)] for k in range(L)]
    P = _bxmm(W, Yc)
    tot = fsum(max(0.0, t) for t in vals)
    return RichResult(payload={
        "components": P,
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "mean": mu,
        "covariance": S,
        "varexplained": [max(0.0, vals[k]) / tot if tot > 0 else 0.0
                         for k in range(L)],
        "mse": fsum(max(0.0, vals[k]) for k in range(L, K)),
        "method": "principal component analysis of signal mixtures by "
                  "eigendecomposition of the covariance matrix, Rangayyan "
                  "Biomedical Signal Analysis 3rd ed. Section 9.7.1, "
                  "eqs. (9.37)-(9.41)",
    })


rangayyan_pca_signals = pcasig  # pre-policy spelling


# -- rgpcaica: Comparative analysis of PCA, ICA, and NMF for signal separation.
def mixcmp(X, ncomp=None, maxiter=200, seed=1):
    """Compare PCA, ICA and NMF as decompositions of the same signal mixture.

    Why: the three matrix decompositions answer different questions of the same
    data -- PCA returns uncorrelated components from second-order statistics,
    ICA returns statistically independent components by exploiting
    non-Gaussianity, and NMF returns nonnegative parts.  Which one to use for
    a given source-separation or signal-analysis job is an empirical question,
    and the way to settle it is to reconstruct with each and measure.
    Rangayyan, *Biomedical Signal Analysis*, 3rd ed., Section 9.7.4, which
    compares exactly these three on decomposition accuracy.

    Each method is run at the requested rank, the mixture is reconstructed from
    its components, and the relative Frobenius reconstruction error is
    reported.  NMF is applied to the nonnegatively shifted mixture, since it
    requires a nonnegative matrix by eq. (9.46).

    Parameters
    ----------
    X : sequence of sequences
        Mixtures, one channel per row.
    ncomp : int, optional
        Common rank; the number of channels by default.
    maxiter : int
        Iteration budget for ICA and NMF.
    seed : int
        Seed for the deterministic initialisations.

    Returns
    -------
    RichResult
        Keys ``error``, ``best``, ``components``, ``rank``, ``method``.
    """
    Y = _bxmat(X, "X")
    K, T = len(Y), len(Y[0])
    L = K if ncomp is None else int(ncomp)
    if not (1 <= L <= K):
        raise ValueError("ncomp must satisfy 1 <= ncomp <= number of channels")
    denom = sqrt(fsum(t * t for r in Y for t in r))
    if denom <= 0.0:
        raise ValueError("X has zero energy")

    def relerr(R):
        return sqrt(fsum((Y[i][t] - R[i][t]) ** 2
                         for i in range(K) for t in range(T))) / denom

    p = pcasig(Y, ncomp=L)
    vecs, mu = p["eigenvectors"], p["mean"]
    B = [[vecs[i][k] for k in range(L)] for i in range(K)]
    Rp = _bxmm(B, p["components"])
    Rp = [[Rp[i][t] + mu[i] for t in range(T)] for i in range(K)]

    ic = icafix(Y, ncomp=L, maxiter=maxiter, seed=seed)
    Ri = _bxmm(ic["mixing"], ic["sources"])
    Ri = [[Ri[i][t] + ic["mean"][i] for t in range(T)] for i in range(K)]

    lo = min(min(r) for r in Y)
    V = [[t - lo for t in r] for r in Y] if lo < 0.0 else [list(r) for r in Y]
    nm = nmfmu(V, L, maxiter=maxiter, seed=seed)
    Rn = _bxmm(nm["W"], nm["H"])
    if lo < 0.0:
        Rn = [[Rn[i][t] + lo for t in range(T)] for i in range(K)]

    err = {"pca": relerr(Rp), "ica": relerr(Ri), "nmf": relerr(Rn)}
    return RichResult(payload={
        "error": err,
        "best": min(err, key=lambda k: err[k]),
        "components": {"pca": p["components"], "ica": ic["sources"],
                       "nmf": {"W": nm["W"], "H": nm["H"]}},
        "rank": L,
        "method": "comparison of PCA, ICA and NMF by relative reconstruction "
                  "error, Rangayyan Biomedical Signal Analysis 3rd ed. "
                  "Section 9.7.4",
    })


rangayyan_pca_vs_ica = mixcmp  # pre-policy spelling


# -- rgppv: Positive predictive value (precision).
def ppv(tp, fp=None, prevalence=None, sensitivity=None,
        specificity=None):
    """Positive predictive value, eq. (10.106).

        PPV = TP / (TP + FP)

    The probability that a subject with a positive test actually has the
    disease.  Unlike the sensitivity and specificity it depends on the
    PREVALENCE, so a PPV measured on a study population where half the
    subjects are ill does not transfer to screening a population where
    one in a thousand is.  Given ``prevalence`` with the sensitivity and
    specificity, the Bayes-corrected value for that population is
    returned alongside.
    """
    if fp is None and tp is not None and not isinstance(tp, (int, float)):
        t = _mat(tp)
        if len(t) != 2 or any(len(r) != 2 for r in t):
            raise ValueError("give TP and FP, or a 2x2 table "
                             "[[TP, FN], [FP, TN]]")
        TP, FP = t[0][0], t[1][0]
    else:
        if fp is None:
            raise ValueError("give TP and FP, or a 2x2 table")
        TP, FP = float(tp), float(fp)
    if TP < 0 or FP < 0:
        raise ValueError("counts cannot be negative")
    n = TP + FP
    if n <= 0:
        raise ValueError("no positive decisions; the PPV is undefined")
    out = {"ppv": TP / n, "precision": TP / n, "tp": TP, "fp": FP,
           "n_positive_calls": n, "depends_on_prevalence": True,
           "method": "Rangayyan (2024) eq. (10.106)"}
    if prevalence is not None:
        if sensitivity is None or specificity is None:
            raise ValueError("a prevalence correction needs both the "
                             "sensitivity and the specificity")
        p = float(prevalence)
        if not 0 <= p <= 1:
            raise ValueError("the prevalence must lie in [0, 1]")
        se, sp = float(sensitivity), float(specificity)
        den = se * p + (1.0 - sp) * (1.0 - p)
        out["prevalence"] = p
        out["ppv_at_prevalence"] = (se * p / den) if den > 0 else None
    return RichResult(payload=out)


rangayyan_ppv = ppv  # pre-policy spelling


# -- rgqda: Quadratic discriminant analysis (QDA) with unequal covariance matrices.
def qda(X, y, query, priors=None):
    """Quadratic discriminant analysis, fitted from data.

        g_k(x) = ln P(C_k) - (1/2) ln|C_k|
                 - (1/2) (x - m_k)^T C_k^-1 (x - m_k)

    eq. (10.73) with the mean and covariance estimated per class, by
    eqs. (10.68)-(10.69).  Because each class keeps its OWN covariance
    the boundaries are quadric surfaces; forcing a single pooled
    covariance is what turns this into linear discriminant analysis, and
    that reduction is reported when it holds.

    A class needs more samples than features or its covariance is
    singular -- QDA estimates p(p+1)/2 covariance parameters PER CLASS,
    so it is the first thing to break on small samples.  That is raised
    rather than worked around.
    """
    Xs = _mat(X)
    ys = list(y)
    q = aslist(query)
    if len(Xs) != len(ys):
        raise ValueError("X and y must have the same number of rows")
    p = len(q)
    if any(len(r) != p for r in Xs):
        raise ValueError("every row of X must match the query length")
    order, grp = _groups(Xs, ys)
    m = len(order)
    if m < 2:
        raise ValueError("need at least two classes")
    for lab in order:
        if len(grp[lab]) <= p:
            raise ValueError("class %r has %d samples for %d features; "
                             "QDA needs more samples than features per "
                             "class or the covariance is singular"
                             % (lab, len(grp[lab]), p))
    if priors is None:
        pr = [len(grp[lab]) / len(Xs) for lab in order]
    else:
        pr = aslist(priors)
        if len(pr) != m:
            raise ValueError("give one prior per class")
    means, covs = [], []
    for lab in order:
        rows = grp[lab]
        mu = _colmeans(rows)
        S = _scatter(rows, mu)
        nk = len(rows)
        covs.append([[S[i][j] / (nk - 1) for j in range(p)]
                     for i in range(p)])
        means.append(mu)
    r = bayesnorm(q, means, covs, priors=pr)
    return RichResult(payload={
        "g": r["d"], "assigned": order[r["assigned"]],
        "assigned_index": r["assigned"], "classes": order,
        "means": means, "covariances": covs, "priors": pr,
        "reduces_to_lda_when_covariances_are_equal":
            r["linear_when_covariances_are_equal"],
        "parameters_per_class": p * (p + 1) // 2,
        "method": "Rangayyan (2024) eqs. (10.68)-(10.73), per-class "
                  "covariances"})


rangayyan_qda = qda  # pre-policy spelling


# -- rgrbf: Radial basis function (RBF) network.
def rbfn(X, y, ncenters=None, spread=1.0, centers=None, ridge=1e-8, query=None):
    """Fit a radial basis function network.

    Why: Cover's theorem says a set of samples that is not linearly separable
    becomes separable once it is projected nonlinearly into a
    higher-dimensional space.  An RBF network does that projection with a layer
    of localised Gaussian responses and then needs only a linear output layer,
    which is why it trains in closed form where a back-propagation network
    needs an iterative search.  This is also the motivation the book gives for
    kernel methods generally.  Rangayyan, *Biomedical Signal Analysis*, 3rd
    ed., Section 10.8.1.

    Network output, eq. (10.86):
        yhat_n = sum_j w_j phi(x_n, c_j) + w_0,
    with the radial basis function of eq. (10.87):
        phi(x_n, c_j) = exp(-log_e(2) ||x_n - c_j||^2 / sigma^2).
    Note the factor log_e(2) in the book's form: phi is exactly one half at
    ||x - c|| = sigma, so ``spread`` reads directly as the half-response
    radius.

    The book notes that picking centres at random gives a needlessly large
    network and that Rangayyan and Wu used orthogonal least squares to choose
    them.  In that spirit the default here is greedy forward selection: centres
    are added one at a time, each time taking the training sample whose
    addition most reduces the residual sum of squares.

    Parameters
    ----------
    X : sequence of sequences
        Feature vectors, one per row.
    y : sequence
        Desired response per row.
    ncenters : int, optional
        Number of hidden neurons J; defaults to min(8, number of samples).
    spread : float
        The spread parameter sigma of eq. (10.87).
    centers : sequence of sequences, optional
        Explicit centres, bypassing the selection step.
    ridge : float
        Ridge term for the output-weight least squares.
    query : sequence of sequences, optional
        Further feature vectors to evaluate.

    Returns
    -------
    RichResult
        Keys ``centers``, ``weights``, ``bias``, ``predictions``,
        ``queryvalues``, ``mse``, ``spread``, ``method``.
    """
    F = _bxmat(X, "X")
    t = _bxvec(y, "y")
    if len(F) != len(t):
        raise ValueError("X and y must have the same number of rows")
    n, p = len(F), len(F[0])
    spread = float(spread)
    if spread <= 0.0:
        raise ValueError("spread must be positive")
    lg2 = log(2.0)

    def phi(a, c):
        d = fsum((a[j] - c[j]) ** 2 for j in range(p))
        return exp(-lg2 * d / (spread * spread))

    if centers is not None:
        Cs = _bxmat(centers, "centers")
        if len(Cs[0]) != p:
            raise ValueError("centers must have the same dimension as X")
    else:
        J = min(8, n) if ncenters is None else int(ncenters)
        if not (1 <= J <= n):
            raise ValueError("ncenters must satisfy 1 <= ncenters <= number of rows")
        chosen, resid = [], list(t)
        for _ in range(J):
            best, bv = -1, -1.0
            for i in range(n):
                if i in chosen:
                    continue
                col = [phi(F[k], F[i]) for k in range(n)]
                den = fsum(c * c for c in col)
                if den <= 1e-14:
                    continue
                v = fsum(col[k] * resid[k] for k in range(n)) ** 2 / den
                if v > bv:
                    best, bv = i, v
            if best < 0:
                break
            chosen.append(best)
            A = [[phi(F[k], F[i]) for i in chosen] + [1.0] for k in range(n)]
            w = _bxlstsq(A, t, float(ridge))
            fit = _bxmv(A, w)
            resid = [t[k] - fit[k] for k in range(n)]
        if not chosen:
            raise ValueError("no usable centre found; check spread and X")
        Cs = [F[i] for i in chosen]

    A = [[phi(F[k], c) for c in Cs] + [1.0] for k in range(n)]
    w = _bxlstsq(A, t, float(ridge))
    fit = _bxmv(A, w)
    mse = fsum((t[k] - fit[k]) ** 2 for k in range(n)) / n

    qv = None
    if query is not None:
        Q = _bxmat(query, "query")
        if len(Q[0]) != p:
            raise ValueError("query must have the same dimension as X")
        qv = [fsum(w[j] * phi(row, Cs[j]) for j in range(len(Cs))) + w[-1]
              for row in Q]

    return RichResult(payload={
        "centers": Cs,
        "weights": w[:-1],
        "bias": w[-1],
        "predictions": fit,
        "queryvalues": qv,
        "mse": mse,
        "spread": spread,
        "method": "radial basis function network with greedy forward centre "
                  "selection, Rangayyan Biomedical Signal Analysis 3rd ed. "
                  "Section 10.8.1, eqs. (10.86)-(10.87)",
    })


rangayyan_rbf_network = rbfn  # pre-policy spelling


# -- rgroc: Receiver operating characteristic (ROC) curve and AUC.
def roc(scores, labels, positive=1):
    """Receiver operating characteristic and its area, Section 10.9.1.

    The ROC traces the sensitivity against 1 - specificity as the
    decision threshold is swept, and the summary the book calls A_z is
    the area under it, bounded in [0, 1].

    The area is computed by the trapezoidal rule over the operating
    points, which for tied scores is exactly the Mann-Whitney statistic
    -- the probability that a randomly chosen diseased subject scores
    above a randomly chosen healthy one, with ties counted as half.
    Summing rectangles instead would silently bias the area whenever
    scores tie, which they do whenever a classifier outputs a class
    rather than a probability.
    """
    s = aslist(scores)
    lab = aslist(labels) if not isinstance(labels[0], str) else list(labels)
    if len(s) != len(lab):
        raise ValueError("scores and labels must have the same length")
    if not s:
        raise ValueError("need at least one observation")
    pos = [s[i] for i in range(len(s)) if lab[i] == positive]
    neg = [s[i] for i in range(len(s)) if lab[i] != positive]
    if not pos or not neg:
        raise ValueError("the ROC needs both classes present")
    npos, nneg = len(pos), len(neg)
    thresholds = sorted(set(s), reverse=True)
    tpf, fpf = [0.0], [0.0]
    for t in thresholds:
        tpf.append(sum(1 for v in pos if v >= t) / npos)
        fpf.append(sum(1 for v in neg if v >= t) / nneg)
    area = 0.0
    for i in range(1, len(fpf)):
        area += 0.5 * (tpf[i] + tpf[i - 1]) * (fpf[i] - fpf[i - 1])
    # Mann-Whitney, ties at one half
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    mw = wins / (npos * nneg)
    # the operating point closest to the top-left corner
    best = min(range(len(tpf)),
               key=lambda i: (1.0 - tpf[i]) ** 2 + fpf[i] ** 2)
    return RichResult(payload={
        "fpf": fpf, "tpf": tpf, "sensitivity": tpf,
        "one_minus_specificity": fpf, "thresholds": thresholds,
        "auc": area, "az": area, "mann_whitney": mw,
        "trapezoidal_equals_mann_whitney": abs(area - mw) < 1e-9,
        "n_positive": npos, "n_negative": nneg,
        "best_index": best,
        "best_operating_point": (fpf[best], tpf[best]),
        "ties_counted_as_half": True,
        "method": "Rangayyan (2024) Section 10.9.1 (ROC, A_z)"})


rangayyan_roc_curve = roc  # pre-policy spelling


# -- rgsapnmf: Sleep apnea diagnosis via NMF of polysomnographic signals.
def ahi(airflow, fs, spo2=None, hours=None, apneafrac=0.10, hypofrac=0.50,
        minsec=10.0, desat=0.0, envsec=1.0):
    """Score sleep apnea from airflow and oximetry: event detection and the AHI.

    Why: the severity of obstructive sleep apnea is reported as one number, the
    apnea-hypopnea index, and that number drives the clinical decision.  It is
    the count of apnea and hypopnea episodes per hour of sleep, so the whole
    scoring problem reduces to detecting episodes and dividing.  Rangayyan,
    *Biomedical Signal Analysis*, 3rd ed., Section 10.13.

    Definitions taken from that section: apnea is the total absence of airflow
    through the mouth and nose; hypopnea is a partial airway collapse that
    makes breathing difficult; an episode must be **at least 10 s** long and be
    linked to a drop in blood oxygenation to be counted; and the severity bands
    are mild 5 to 15 events/h, moderate 15 to 30 events/h, and severe above 30
    events/h.

    The book gives no numeric amplitude threshold separating apnea from
    hypopnea, nor a numeric desaturation threshold, so ``apneafrac``,
    ``hypofrac`` and ``desat`` are explicit parameters rather than hidden
    constants; the defaults (10% and 50% of the running baseline amplitude, any
    drop in SpO2) are stated here and are not attributed to Rangayyan.

    The amplitude envelope is a running maximum over +/- ``envsec`` seconds,
    which must span a breath cycle to be stable but which also shortens every
    detected episode by about 2 * ``envsec``.  Since the 10 s minimum is a hard
    clinical criterion, ``envsec`` is exposed rather than fixed: raise it for
    slow breathing, lower it if short episodes are being missed.

    Parameters
    ----------
    airflow : sequence
        Airflow signal.
    fs : float
        Sampling rate in Hz.
    spo2 : sequence, optional
        Blood oxygen saturation on the same time base.  Without it the
        oxygenation condition cannot be checked and every amplitude event is
        counted; the payload records that.
    hours : float, optional
        Hours of sleep; the recording duration by default.
    apneafrac, hypofrac : float
        Envelope thresholds, as fractions of the baseline amplitude.
    minsec : float
        Minimum episode duration; the book requires at least 10 s.
    desat : float
        Required drop in SpO2, in the units of ``spo2``.

    Returns
    -------
    RichResult
        Keys ``ahi``, ``severity``, ``apnea``, ``hypopnea``, ``events``,
        ``hours``, ``oxygenchecked``, ``method``.
    """
    x = _bxvec(airflow, "airflow")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be a positive sampling rate in Hz")
    minsec = float(minsec)
    if minsec <= 0.0:
        raise ValueError("minsec must be positive")
    if not (0.0 < apneafrac < hypofrac <= 1.0):
        raise ValueError("need 0 < apneafrac < hypofrac <= 1")
    dur = len(x) / fs
    hrs = dur / 3600.0 if hours is None else float(hours)
    if hrs <= 0.0:
        raise ValueError("hours must be positive")
    envsec = float(envsec)
    if envsec <= 0.0:
        raise ValueError("envsec must be positive")
    ox = None
    if spo2 is not None:
        ox = _bxvec(spo2, "spo2")
        if len(ox) != len(x):
            raise ValueError("spo2 must have the same length as airflow")

    w = max(1, int(round(envsec * fs)))
    env = []
    for i in range(len(x)):
        lo, hi = max(0, i - w), min(len(x), i + w + 1)
        env.append(max(abs(t) for t in x[lo:hi]))
    base = sorted(env)[int(0.75 * (len(env) - 1))]
    if base <= 0.0:
        raise ValueError("airflow has no measurable amplitude")

    need = int(round(minsec * fs))
    events = []
    i = 0
    while i < len(env):
        if env[i] < hypofrac * base:
            j = i
            while j < len(env) and env[j] < hypofrac * base:
                j += 1
            if j - i >= need:
                seg = env[i:j]
                kind = "apnea" if min(seg) < apneafrac * base else "hypopnea"
                ok = True
                dv = None
                if ox is not None:
                    pre = ox[max(0, i - int(fs * 30)):i + 1] or ox[i:i + 1]
                    dv = max(pre) - min(ox[i:j])
                    ok = dv > float(desat)
                if ok:
                    events.append({"kind": kind, "start": i / fs, "end": j / fs,
                                   "duration": (j - i) / fs, "desaturation": dv})
            i = j
        else:
            i += 1

    na = sum(1 for e in events if e["kind"] == "apnea")
    nh = len(events) - na
    index = (na + nh) / hrs
    if index < 5.0:
        sev = "normal"
    elif index < 15.0:
        sev = "mild"
    elif index < 30.0:
        sev = "moderate"
    else:
        sev = "severe"

    return RichResult(payload={
        "ahi": index,
        "severity": sev,
        "apnea": na,
        "hypopnea": nh,
        "events": events,
        "hours": hrs,
        "baseline": base,
        "envsec": envsec,
        "oxygenchecked": ox is not None,
        "method": "apnea-hypopnea index from airflow and oximetry with the "
                  "10 s minimum episode duration and the mild/moderate/severe "
                  "bands of Rangayyan Biomedical Signal Analysis 3rd ed. "
                  "Section 10.13; amplitude and desaturation thresholds are "
                  "parameters, not values given by that section",
    })


rangayyan_sleep_apnea_nmf = ahi  # pre-policy spelling


# -- rgsen: Sensitivity (recall, true positive rate).
def sens(tp, fn=None):
    """Sensitivity, eq. (10.100).

        S+ = (number of TP decisions) / (number of subjects with the
                                         disease)

    Also the true-positive fraction.  It measures the capability to
    DETECT the disease and says nothing at all about false alarms, so a
    test that calls everyone positive scores a perfect 1 -- which is why
    the book always reports it beside the specificity.

    Pass the counts, or a 2x2 table [[TP, FN], [FP, TN]].
    """
    if fn is None:
        t = _mat(tp)
        if len(t) != 2 or any(len(r) != 2 for r in t):
            raise ValueError("give TP and FN, or a 2x2 table "
                             "[[TP, FN], [FP, TN]]")
        TP, FN = t[0][0], t[0][1]
    else:
        TP, FN = float(tp), float(fn)
    if TP < 0 or FN < 0:
        raise ValueError("counts cannot be negative")
    n = TP + FN
    if n <= 0:
        raise ValueError("no subjects with the disease; the sensitivity "
                         "is undefined")
    return RichResult(payload={
        "sensitivity": TP / n, "tpf": TP / n, "fnf": FN / n,
        "n_diseased": n, "tp": TP, "fn": FN,
        "says_nothing_about_false_alarms": True,
        "method": "Rangayyan (2024) eq. (10.100)"})


rangayyan_sensitivity = sens  # pre-policy spelling


# -- rgsepix: Separability index: ratio of between-class to within-class scatter.
def sepindex(X, y):
    """Separability index from the scatter matrices, Section 10.10.1.

        J = tr(S_B) / tr(S_W)

    with S_W the within-class scatter, summed over classes, and S_B the
    between-class scatter of the class means about the grand mean,
    weighted by class size.  Larger is better separated.

    The trace ratio ignores the OFF-diagonal structure of both matrices,
    so it cannot see that a pair of features is jointly discriminating
    when neither is discriminating alone -- for that the book's
    divergence, which uses the full covariance, is the measure to reach
    for.
    """
    Xs = _mat(X)
    ys = list(y)
    if len(Xs) != len(ys):
        raise ValueError("X and y must have the same number of rows")
    if len(Xs) < 2:
        raise ValueError("need at least two samples")
    p = len(Xs[0])
    if any(len(r) != p for r in Xs):
        raise ValueError("every row of X must have the same length")
    order, grp = _groups(Xs, ys)
    if len(order) < 2:
        raise ValueError("need at least two classes")
    grand = _colmeans(Xs)
    SW = [[0.0] * p for _ in range(p)]
    SB = [[0.0] * p for _ in range(p)]
    for lab in order:
        rows = grp[lab]
        mu = _colmeans(rows)
        s = _scatter(rows, mu)
        for i in range(p):
            for j in range(p):
                SW[i][j] += s[i][j]
        d = [mu[i] - grand[i] for i in range(p)]
        for i in range(p):
            for j in range(p):
                SB[i][j] += len(rows) * d[i] * d[j]
    tw = _trace(SW)
    if tw <= 0:
        raise ValueError("the within-class scatter vanishes; every class "
                         "is a single repeated point")
    return RichResult(payload={
        "j": _trace(SB) / tw, "trace_between": _trace(SB),
        "trace_within": tw, "s_within": SW, "s_between": SB,
        "classes": order, "n_classes": len(order), "n_features": p,
        "ignores_off_diagonal_structure": True,
        "method": "Rangayyan (2024) Section 10.10.1 (separability of "
                  "features)"})


rangayyan_separability_index = sepindex  # pre-policy spelling


# -- rgspe: Specificity (true negative rate).
def spec(tn, fp=None):
    """Specificity, eq. (10.101).

        S- = (number of TN decisions) / (number of subjects without the
                                         disease)

    The true-negative fraction, the mirror of the sensitivity: it
    measures accuracy in identifying the ABSENCE of the disease.  The
    book's identities S- = 1 - FPF and S+ = 1 - FNF are returned so the
    complementary fractions need not be recomputed.

    Pass the counts, or a 2x2 table [[TP, FN], [FP, TN]].
    """
    if fp is None:
        t = _mat(tn)
        if len(t) != 2 or any(len(r) != 2 for r in t):
            raise ValueError("give TN and FP, or a 2x2 table "
                             "[[TP, FN], [FP, TN]]")
        TN, FP = t[1][1], t[1][0]
    else:
        TN, FP = float(tn), float(fp)
    if TN < 0 or FP < 0:
        raise ValueError("counts cannot be negative")
    n = TN + FP
    if n <= 0:
        raise ValueError("no subjects without the disease; the "
                         "specificity is undefined")
    return RichResult(payload={
        "specificity": TN / n, "tnf": TN / n, "fpf": FP / n,
        "n_healthy": n, "tn": TN, "fp": FP,
        "method": "Rangayyan (2024) eq. (10.101)"})


rangayyan_specificity = spec  # pre-policy spelling


# -- rgsprep: Sparse representation of biomedical signals in learned dictionary.
def sparsecode(x, D, sparsity=None, lam=None, maxiter=2000, tol=1e-10):
    """Represent one signal sparsely in a dictionary, by count or by penalty.

    Why: the point of an overcomplete dictionary is that only a handful of its
    atoms are needed for any one signal, and those few coefficients are a far
    more compact and more discriminative description than the raw samples.
    There are two ways to ask for "few": bound the number of atoms outright, or
    penalise the sum of the coefficient magnitudes and let the solution decide.
    Rangayyan, *Biomedical Signal Analysis*, 3rd ed., Section 9.5 discusses the
    greedy route and calls it a greedy approximation, choosing the best option
    available at each step without regard to the final outcome.

    ``sparsity=T``  solves min ||x - D' a||^2 subject to ||a||_0 <= T by
    orthogonal matching pursuit (Pati, Rezaiifar and Krishnaprasad, Asilomar
    1993).  ``lam``  solves min 0.5 ||x - D' a||^2 + lam ||a||_1, the lasso of
    Tibshirani, Journal of the Royal Statistical Society Series B
    58(1):267-288, 1996, by iterative soft thresholding.  Neither the lasso nor
    the orthogonalised pursuit is presented by Rangayyan; only the plain
    matching pursuit of Section 9.3 is.

    Parameters
    ----------
    x : sequence
        Signal to represent.
    D : sequence of sequences
        Dictionary atoms, one per row.
    sparsity : int, optional
        Atom budget T.  Exactly one of ``sparsity`` and ``lam`` is required.
    lam : float, optional
        L1 penalty weight.
    maxiter : int
        Iteration budget for the penalised route.
    tol : float
        Convergence tolerance.

    Returns
    -------
    RichResult
        Keys ``alpha``, ``support``, ``reconstruction``, ``residual``,
        ``error``, ``energyratio``, ``mode``, ``method``.
    """
    if (sparsity is None) == (lam is None):
        raise ValueError("give exactly one of sparsity (atom budget) or "
                         "lam (L1 penalty)")
    x = _bxvec(x, "x")
    A = _bxmat(D, "D")
    n = len(x)
    if any(len(a) != n for a in A):
        raise ValueError("every dictionary atom must have the same length as x")
    e0 = fsum(t * t for t in x)
    if e0 <= 0.0:
        raise ValueError("x has zero energy")

    if sparsity is not None:
        r = ompfit(x, A, sparsity=int(sparsity), tol=float(tol))
        mode = "omp"
        src = ("orthogonal matching pursuit; Pati, Rezaiifar and "
               "Krishnaprasad, Asilomar 1993")
    else:
        r = bpursuit(x, A, lam=float(lam), maxiter=int(maxiter), tol=float(tol))
        mode = "lasso"
        src = ("lasso by iterative soft thresholding; Tibshirani, JRSS B "
               "58(1):267-288, 1996")

    a = r["alpha"] if mode == "lasso" else r["coefficients"]
    rec = r["reconstruction"]
    res = r["residual"]
    err = _bxnrm(res)
    return RichResult(payload={
        "alpha": a,
        "support": [j for j in range(len(A)) if a[j] != 0.0],
        "reconstruction": rec,
        "residual": res,
        "error": err,
        "energyratio": 1.0 - (err * err) / e0,
        "mode": mode,
        "method": "sparse representation of a biomedical signal in a learned "
                  "dictionary, in the greedy-approximation framing of "
                  "Rangayyan Biomedical Signal Analysis 3rd ed. Section 9.5; "
                  "solver: " + src,
    })


rangayyan_sparse_rep = sparsecode  # pre-policy spelling


# -- rgsvm: Support vector machine (SVM) via margin maximization.
def svm(X, y, C=1.0, maxiter=2000, tol=1e-6):
    """Linear support vector machine, Section 10.4.5.

    Maximizes the margin 2/||w|| subject to y_i (w^T x_i + b) >= 1, in
    the dual

        max sum a_i - (1/2) sum_i sum_j a_i a_j y_i y_j x_i^T x_j
        s.t. 0 <= a_i <= C,  sum a_i y_i = 0

    solved by SMO-style coordinate ascent on pairs, which respects the
    equality constraint that single-coordinate updates cannot.

    Only the patterns with a_i > 0 -- the SUPPORT VECTORS -- enter the
    solution, so the boundary is set by the samples nearest it and is
    untouched by the bulk of the data.  That is the SVM's strength on
    small samples and its weakness against a single mislabelled point
    near the boundary, which C controls: a small C tolerates violations,
    a large one insists on separating and will contort the boundary
    around an outlier.

    Labels must be -1 and +1.
    """
    Xs = _mat(X)
    ys = [float(v) for v in y]
    n = len(Xs)
    if n != len(ys):
        raise ValueError("X and y must have the same number of rows")
    if set(ys) - {-1.0, 1.0}:
        raise ValueError("the SVM needs labels -1 and +1")
    if len(set(ys)) < 2:
        raise ValueError("both classes must be present")
    p = len(Xs[0])
    Cv = float(C)
    if Cv <= 0:
        raise ValueError("C must be positive")
    K = [[fsum(Xs[i][t] * Xs[j][t] for t in range(p)) for j in range(n)]
         for i in range(n)]
    a = [0.0] * n
    b = 0.0
    it = 0
    for it in range(1, int(maxiter) + 1):
        changed = 0
        for i in range(n):
            fi = fsum(a[t] * ys[t] * K[t][i] for t in range(n)) + b
            Ei = fi - ys[i]
            if (ys[i] * Ei < -tol and a[i] < Cv) or \
               (ys[i] * Ei > tol and a[i] > 0):
                j = (i + 1 + it) % n
                if j == i:
                    continue
                fj = fsum(a[t] * ys[t] * K[t][j] for t in range(n)) + b
                Ej = fj - ys[j]
                ai, aj = a[i], a[j]
                if ys[i] != ys[j]:
                    L, H = max(0.0, aj - ai), min(Cv, Cv + aj - ai)
                else:
                    L, H = max(0.0, ai + aj - Cv), min(Cv, ai + aj)
                if H - L < 1e-12:
                    continue
                eta = 2.0 * K[i][j] - K[i][i] - K[j][j]
                if eta >= -1e-12:
                    continue
                anj = aj - ys[j] * (Ei - Ej) / eta
                anj = min(H, max(L, anj))
                if abs(anj - aj) < 1e-12:
                    continue
                ani = ai + ys[i] * ys[j] * (aj - anj)
                b1 = b - Ei - ys[i] * (ani - ai) * K[i][i] \
                    - ys[j] * (anj - aj) * K[i][j]
                b2 = b - Ej - ys[i] * (ani - ai) * K[i][j] \
                    - ys[j] * (anj - aj) * K[j][j]
                if 0 < ani < Cv:
                    b = b1
                elif 0 < anj < Cv:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
                a[i], a[j] = ani, anj
                changed += 1
        if changed == 0:
            break
    w = [fsum(a[i] * ys[i] * Xs[i][t] for i in range(n))
         for t in range(p)]
    sv = [i for i in range(n) if a[i] > 1e-8]
    marg = (2.0 / sqrt(fsum(v * v for v in w))) if any(w) else float("inf")
    pred = [1.0 if fsum(w[t] * Xs[i][t] for t in range(p)) + b >= 0
            else -1.0 for i in range(n)]
    acc = sum(1 for i in range(n) if pred[i] == ys[i]) / n
    return RichResult(payload={
        "w": w, "b": b, "alpha": a, "support_vectors": sv,
        "n_support": len(sv), "margin": marg, "C": Cv,
        "iterations": it, "converged": it < int(maxiter),
        "training_accuracy": acc,
        "boundary_set_by_the_support_vectors_only": True,
        "large_c_contorts_around_outliers": True,
        "method": "Rangayyan (2024) Section 10.4.5 (support vector "
                  "machine)"})


rangayyan_svm = svm  # pre-policy spelling


# -- rgsvmk: SVM with kernel trick (RBF, polynomial, sigmoid kernels).
def svmkern(X, y, query=None, kernel="rbf", gamma=None, degree=3,
            coef0=0.0, C=1.0, maxiter=2000, tol=1e-6):
    """Kernel SVM, Section 10.4.5.

        K(x, x') = exp(-gamma ||x - x'||^2)      RBF
                 = (x^T x' + coef0)^degree       polynomial
                 = tanh(gamma x^T x' + coef0)    sigmoid

    The dual of the linear SVM depends on the data only through inner
    products, so replacing that product with a kernel fits a linear
    boundary in a space the data is never actually mapped into -- the
    kernel trick.  The boundary in the ORIGINAL space is then curved, and
    there is no weight vector to report: the classifier is the list of
    support vectors and their coefficients, which is why kernel SVMs grow
    with the training set where the linear one does not.

    gamma defaults to 1/n_features.  The sigmoid kernel is not positive
    definite for all parameters, so the dual is not guaranteed concave
    and the solution may depend on the starting point; that is reported
    rather than assumed away.
    """
    Xs = _mat(X)
    ys = [float(v) for v in y]
    n = len(Xs)
    if n != len(ys):
        raise ValueError("X and y must have the same number of rows")
    if set(ys) - {-1.0, 1.0}:
        raise ValueError("the SVM needs labels -1 and +1")
    p = len(Xs[0])
    g = (1.0 / p) if gamma is None else float(gamma)
    if kernel not in ("rbf", "poly", "linear", "sigmoid"):
        raise ValueError("kernel must be 'rbf', 'poly', 'linear' or "
                         "'sigmoid'")

    def kf(u, v):
        dot = fsum(u[t] * v[t] for t in range(p))
        if kernel == "linear":
            return dot
        if kernel == "poly":
            return (dot + float(coef0)) ** int(degree)
        if kernel == "sigmoid":
            return _tanh(g * dot + float(coef0))
        d2 = fsum((u[t] - v[t]) ** 2 for t in range(p))
        return exp(-g * d2)

    K = [[kf(Xs[i], Xs[j]) for j in range(n)] for i in range(n)]
    Cv = float(C)
    a = [0.0] * n
    b = 0.0
    it = 0
    for it in range(1, int(maxiter) + 1):
        changed = 0
        for i in range(n):
            fi = fsum(a[t] * ys[t] * K[t][i] for t in range(n)) + b
            Ei = fi - ys[i]
            if (ys[i] * Ei < -tol and a[i] < Cv) or \
               (ys[i] * Ei > tol and a[i] > 0):
                j = (i + 1 + it) % n
                if j == i:
                    continue
                fj = fsum(a[t] * ys[t] * K[t][j] for t in range(n)) + b
                Ej = fj - ys[j]
                ai, aj = a[i], a[j]
                if ys[i] != ys[j]:
                    L, H = max(0.0, aj - ai), min(Cv, Cv + aj - ai)
                else:
                    L, H = max(0.0, ai + aj - Cv), min(Cv, ai + aj)
                if H - L < 1e-12:
                    continue
                eta = 2.0 * K[i][j] - K[i][i] - K[j][j]
                if eta >= -1e-12:
                    continue
                anj = min(H, max(L, aj - ys[j] * (Ei - Ej) / eta))
                if abs(anj - aj) < 1e-12:
                    continue
                ani = ai + ys[i] * ys[j] * (aj - anj)
                b1 = b - Ei - ys[i] * (ani - ai) * K[i][i] \
                    - ys[j] * (anj - aj) * K[i][j]
                b2 = b - Ej - ys[i] * (ani - ai) * K[i][j] \
                    - ys[j] * (anj - aj) * K[j][j]
                b = b1 if 0 < ani < Cv else (b2 if 0 < anj < Cv
                                             else 0.5 * (b1 + b2))
                a[i], a[j] = ani, anj
                changed += 1
        if changed == 0:
            break
    sv = [i for i in range(n) if a[i] > 1e-8]
    pred = [1.0 if fsum(a[t] * ys[t] * K[t][i] for t in range(n)) + b >= 0
            else -1.0 for i in range(n)]
    acc = sum(1 for i in range(n) if pred[i] == ys[i]) / n
    out = {"alpha": a, "b": b, "support_vectors": sv,
           "n_support": len(sv), "kernel": kernel, "gamma": g, "C": Cv,
           "iterations": it, "converged": it < int(maxiter),
           "training_accuracy": acc,
           "no_weight_vector_in_the_original_space": kernel != "linear",
           "model_grows_with_the_training_set": True,
           "sigmoid_kernel_is_not_always_positive_definite":
               kernel == "sigmoid",
           "method": "Rangayyan (2024) Section 10.4.5 (kernel SVM)"}
    if query is not None:
        q = aslist(query)
        if len(q) != p:
            raise ValueError("the query must match the feature length")
        s = fsum(a[t] * ys[t] * kf(Xs[t], q) for t in range(n)) + b
        out["decision"] = s
        out["assigned"] = 1.0 if s >= 0 else -1.0
    return RichResult(payload=out)


rangayyan_svm_kernel = svmkern  # pre-policy spelling


# -- rgvagadp: Adaptive TFD of VAG signals via matching pursuit.
def vagtfd(x, fs, natoms=12, nfreq=32, ntime=None, lag=12):
    """Adaptive time-frequency distribution of a VAG signal and its four features.

    Why: bilinear time-frequency distributions buy resolution at the cost of
    cross-terms between signal components.  If the signal is first decomposed
    into known components, the interaction between them is known too, and the
    cross-terms can simply be left out of the sum -- which is what makes a
    decomposition-based TFD adaptive rather than merely smoothed.  Rangayyan,
    *Biomedical Signal Analysis*, 3rd ed., Section 9.6, applied to knee-joint
    vibroarthrography in Section 9.9.

    The signal is decomposed by matching pursuit (Section 9.3), and the TFD is
    the energy-weighted sum of the Wigner distributions of the selected atoms,
    the first term of eq. (9.15); the cross-term double sum of that equation is
    omitted, which is exactly the cross-term removal the adaptive TFD is for.
    Four time-varying features are then taken from the distribution M(t, w),
    Section 9.9:

    * EP, eq. (9.79), the mean of M along each time slice: energy against time;
    * ESP, eq. (9.80), the standard deviation along the slice: spread of energy
      over frequency, higher for the multicomponent signals that rough,
      nonuniform cartilage produces;
    * FP, eq. (9.81), the first moment along the slice: instantaneous mean
      frequency;
    * FSP, eq. (9.82), the second central moment: spread about that mean
      frequency, sensitive to amplitude modulation.

    Parameters
    ----------
    x : sequence
        VAG signal.
    fs : float
        Sampling rate in Hz.
    natoms : int
        Number of matching-pursuit atoms M.
    nfreq : int
        Number of frequency bins of the distribution.
    ntime : int, optional
        Number of time slices; every sample by default.
    lag : int
        Half-width of the lag window of the discrete Wigner distribution.

    Returns
    -------
    RichResult
        Keys ``tfd``, ``times``, ``frequencies``, ``ep``, ``esp``, ``fp``,
        ``fsp``, ``coefficients``, ``method``.
    """
    x = _bxvec(x, "x")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be a positive sampling rate in Hz")
    n = len(x)
    nfreq = int(nfreq)
    lag = int(lag)
    if nfreq < 2 or lag < 1:
        raise ValueError("need nfreq >= 2 and lag >= 1")
    nt = n if ntime is None else int(ntime)
    if not (1 <= nt <= n):
        raise ValueError("ntime must satisfy 1 <= ntime <= len(x)")

    mp = mpursuit(x, natoms=natoms)
    coef, atoms = mp["coefficients"], mp["atoms"]
    if not atoms:
        raise ValueError("matching pursuit selected no atoms")

    tidx = [int(round(k * (n - 1) / (nt - 1))) if nt > 1 else 0 for k in range(nt)]
    tfd = [[0.0] * nfreq for _ in range(nt)]
    for a_i in range(len(atoms)):
        g = atoms[a_i]
        w2 = coef[a_i] ** 2
        for ti in range(nt):
            c = tidx[ti]
            prod = []
            for m in range(-lag, lag + 1):
                p, q = c + m, c - m
                prod.append(g[p] * g[q] if 0 <= p < n and 0 <= q < n else 0.0)
            for fi in range(nfreq):
                th = -2.0 * pi * fi / (2.0 * nfreq)
                acc = fsum(prod[m + lag] * cos(th * 2 * m) for m in range(-lag, lag + 1))
                tfd[ti][fi] += w2 * 2.0 * acc

    freqs = [fi * fs / (2.0 * nfreq) for fi in range(nfreq)]
    times = [t / fs for t in tidx]
    ep, esp, fp, fsp = [], [], [], []
    for ti in range(nt):
        row = tfd[ti]
        e = fsum(row) / nfreq
        ep.append(e)
        esp.append(sqrt(max(0.0, fsum((t - e) ** 2 for t in row) / nfreq)))
        pos = [max(0.0, t) for t in row]
        s = fsum(pos)
        if s > 0.0:
            f1 = fsum(freqs[k] * pos[k] for k in range(nfreq)) / s
            fp.append(f1)
            fsp.append(sqrt(max(0.0, fsum((freqs[k] - f1) ** 2 * pos[k]
                                          for k in range(nfreq)) / s)))
        else:
            fp.append(0.0)
            fsp.append(0.0)

    return RichResult(payload={
        "tfd": tfd,
        "times": times,
        "frequencies": freqs,
        "ep": ep,
        "esp": esp,
        "fp": fp,
        "fsp": fsp,
        "coefficients": coef,
        "method": "matching-pursuit adaptive TFD of a VAG signal with the "
                  "EP/ESP/FP/FSP features, Rangayyan Biomedical Signal "
                  "Analysis 3rd ed. Section 9.6, eq. (9.15) diagonal term, "
                  "and Section 9.9, eqs. (9.79)-(9.82)",
    })


rangayyan_vag_adaptive_tfd = vagtfd  # pre-policy spelling


# -- rng190: Pan-Tompkins peak classification.
def rangayyan_ch4_pan_tompkins_peak_classification(PEAKI, SPKI=None, NPKI=None,
                                                   is_signal=None):
    r"""Pan-Tompkins adaptive threshold update (Rangayyan Ch. 4):

    .. math:: SPKI &= 0.125\,PEAKI + 0.875\,SPKI
              \quad\text{(signal peak)}\\
              NPKI &= 0.125\,PEAKI + 0.875\,NPKI
              \quad\text{(noise peak)}

    Two exponential trackers with the SAME 1/8 coefficient, updated
    according to which class the peak was assigned. The detection
    threshold is :math:`NPKI + 0.25(SPKI - NPKI)`, which floats
    between the two estimates so the detector adapts to changing
    amplitude without a fixed cutoff.

    Parameters
    ----------
    PEAKI : float or array-like
        Peak amplitude(s), processed in order.
    SPKI, NPKI : float, optional
        Running signal and noise estimates; initialised from the first
        peak when omitted.
    is_signal : bool or array-like of bool, optional
        Class of each peak; peaks above the current threshold are
        treated as signal when omitted.

    Returns
    -------
    RichResult
        keys: ``SPKI``, ``NPKI``, ``threshold``, ``classified``
        (per peak), ``n_peaks``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4 (the Pan-Tompkins algorithm).
    """
    peaks = np.atleast_1d(np.asarray(PEAKI, dtype=float))
    if peaks.size < 1:
        raise ValueError("PEAKI must be non-empty.")
    if np.any(peaks < 0):
        raise ValueError("peak amplitudes must be non-negative.")
    spki = float(peaks[0]) if SPKI is None else float(SPKI)
    npki = float(peaks[0]) / 2.0 if NPKI is None else float(NPKI)
    flags = None
    if is_signal is not None:
        flags = np.atleast_1d(np.asarray(is_signal, dtype=bool))
        if flags.size != peaks.size:
            raise ValueError("is_signal must have one entry per peak.")
    classified = []
    for i, p in enumerate(peaks):
        thr = npki + 0.25 * (spki - npki)
        sig = bool(p > thr) if flags is None else bool(flags[i])
        if sig:
            spki = 0.125 * p + 0.875 * spki
        else:
            npki = 0.125 * p + 0.875 * npki
        classified.append(sig)
    return RichResult(payload={"SPKI": spki, "NPKI": npki,
                               "threshold": npki + 0.25 * (spki - npki),
                               "classified": np.array(classified),
                               "n_peaks": int(peaks.size),
                               "method": "Pan-Tompkins 1/8 trackers; threshold floats between them"})


_CHEATSHEET = [
    'rgacc: Classification accuracy.',
    'Two-layer perceptron trained by back-propagation (Rangayyan eqs. 10.79-10.85).',
    'rgbayes: Bayes minimum-error classifier.',
    'rgbayng: Bayes classifier for normal (Gaussian) patterns.',
    'Incomplete left/right bundle-branch block decision rules (Rangayyan Section 10.2.1).',
    'Bayes classifier for normal vs. ectopic beats on [QRSTA, FF] (Rangayyan Section 10.11.2).',
    'NMF channel selection and weighting for a motor-imagery BCI (Rangayyan eqs. 9.94-9.96).',
    'rgbhatt: Bhattacharyya distance for class separability.',
    'Basis-pursuit denoising by iterative soft thresholding (Chen, Donoho and Saunders 1998).',
    'Cross-validated CAD pipeline scored by sensitivity/specificity/accuracy (Rangayyan Ch. 10).',
    '1-D CNN forward pass: convolution, rectifier, max-pooling, softmax readout.',
    'Single-channel fetal ECG extraction by NMF of the STFT magnitude (Rangayyan Section 9.11).',
    'Linear discriminant on [RR, form factor] for normal vs. PVC beats (Rangayyan eq. 10.131).',
    'Fractional EEG power in the delta/theta/alpha/beta/gamma bands (Rangayyan Section 1.2.6).',
    'rgelbow: Elbow method for k-means cluster count selection.',
    'Seizure detection by signal-derived dictionary learning, Algorithm 9.2 (Rangayyan Section 9.8).',
    'rgerrbd: Bhattacharyya bound on Bayes classification error.',
    "rgfish: Fisher's criterion for feature separability.",
    'rgfld: Fisher linear discriminant analysis (LDA).',
    'rghier: Hierarchical agglomerative clustering.',
    'FastICA fixed-point independent component analysis with tanh nonlinearity.',
    'EEG artifact removal by zeroing high-kurtosis ICA components and back-projecting.',
    'Infomax ICA with the natural-gradient update (Bell and Sejnowski 1995).',
    'rgkfcv: K-fold cross-validation.',
    'rgkmns: K-means clustering algorithm.',
    'Knee VAG cartilage screening: variance of segment means and the two-step duration rule.',
    'rgknn: K-nearest neighbor (k-NN) classifier.',
    'K-SVD dictionary learning with OMP sparse coding (Aharon, Elad and Bruckstein 2006).',
    'Sparse-code a signal set against a fixed dictionary by orthogonal matching pursuit.',
    'rglindf: Linear discriminant function for pattern classification.',
    'rglindsep: Linear discriminant function with optimal separability.',
    'rgloo: Leave-one-out cross-validation (LOO-CV).',
    'rglr: Logistic regression for binary classification.',
    'LSTM recurrence with a ridge least-squares readout on the final hidden state.',
    'rgmahd: Mahalanobis distance from sample to class.',
    "rgmcn: McNemar's test for comparing two classifiers.",
    'Matching-pursuit decomposition into Gabor time-frequency atoms (Rangayyan eqs. 9.1-9.7).',
    'Kalman-filter neural decoder for prosthesis control (Rangayyan Section 8.18, eqs. 8.95-8.99).',
    'Nonnegative matrix factorisation by multiplicative updates (Rangayyan eqs. 9.49, 9.50).',
    'Rank EEG channels by the normalised NMF basis-row RMS deviation (Rangayyan eqs. 9.94-9.96).',
    'Orthogonal matching pursuit with least-squares reprojection (Pati et al. 1993).',
    'PCA of correlated signals by eigendecomposition of the covariance (Rangayyan eqs. 9.37-9.41).',
    'Compare PCA, ICA and NMF on one mixture by reconstruction error (Rangayyan Section 9.7.4).',
    'rgppv: Positive predictive value (precision).',
    'rgqda: Quadratic discriminant analysis (QDA) with unequal covariance matrices.',
    'RBF network with greedy centre selection and closed-form weights (Rangayyan eqs. 10.86-10.87).',
    'rgroc: Receiver operating characteristic (ROC) curve and AUC.',
    'Apnea-hypopnea index and severity from airflow and oximetry (Rangayyan Section 10.13).',
    'rgsen: Sensitivity (recall, true positive rate).',
    'rgsepix: Separability index: ratio of between-class to within-class scatter.',
    'rgspe: Specificity (true negative rate).',
    'Sparse representation of one signal by OMP (atom budget) or lasso (L1 penalty).',
    'rgsvm: Support vector machine (SVM) via margin maximization.',
    'rgsvmk: SVM with kernel trick (RBF, polynomial, sigmoid kernels).',
    'MP-based adaptive TFD of a VAG signal with the EP/ESP/FP/FSP features (Rangayyan 9.6, 9.9).',
    'rng190: Pan-Tompkins peak classification.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)

# Pre-policy run-together spellings.  These were in the lazy
# map but not in the module, so morie.fn.<name> raised
# AttributeError.  Restored rather than dropped, because the
# map is the public flat namespace.
rangayyanksvd = rangayyan_ksvd  # pre-policy spelling, kept live
rangayyanloocv = loocv  # pre-policy spelling, kept live
rangayyannmf = rangayyan_nmf  # pre-policy spelling, kept live
rangayyanomp = rangayyan_omp  # pre-policy spelling, kept live
rangayyanppv = rangayyan_ppv  # pre-policy spelling, kept live
rangayyanqda = rangayyan_qda  # pre-policy spelling, kept live
rangayyansvm = rangayyan_svm  # pre-policy spelling, kept live
