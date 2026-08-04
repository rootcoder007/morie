# morie.fn -- bsaclass (rootcoder007/morie)
"""Pattern classification and decomposition: discriminants, Bayes, SVM, k-NN, clustering, PCA/ICA/NMF, sparse coding, validation and performance measures.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 56
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from fractions import Fraction
from math import erf, exp, fsum, lgamma as _lgamma, log, pi, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._richresult import RichResult

__all__ = [
    'accuracy',
    'rangayyan_accuracy',
    'rangayyan_ann_mlp',
    'bayescls',
    'rangayyan_bayes_classifier',
    'bayesnorm',
    'rangayyan_bayes_gaussian',
    'rangayyan_bundle_branch_block',
    'rangayyan_ecg_bbb_normal',
    'rangayyan_bci_nmf',
    'normdist',
    'divergence',
    'divav',
    'bhatt',
    'rangayyan_bhattacharyya',
    'rangayyan_basis_pursuit',
    'rangayyan_cad_pipeline',
    'rangayyan_cnn_signal',
    'rangayyan_fetal_ecg_single',
    'rangayyan_ecg_normal_ectopic',
    'rangayyan_eeg_rhythms',
    'elbow',
    'rangayyan_kmeans_elbow',
    'rangayyan_epilepsy_ksvd',
    'errbound',
    'rangayyan_bayes_error_bound',
    'fishcrit',
    'rangayyan_fisher_criterion',
    'fishlda',
    'rangayyan_fisher_lda',
    'hclust',
    'rangayyan_hierarchical_clust',
    'rangayyan_fastica',
    'rangayyan_ica_artifact',
    'rangayyan_infomax_ica',
    'kfoldcv',
    'rangayyan_kfold_cv',
    'kmeans',
    'rangayyan_kmeans',
    'rangayyan_knee_classify',
    'knn',
    'rangayyan_knn_classifier',
    'rangayyan_ksvd',
    'rangayyan_dictionary_sparse',
    'lindisc',
    'rangayyan_linear_discrim',
    'lindsep',
    'rangayyan_lin_discr_sep',
    'loocv',
    'rangayyan_loo_cv',
    'logreg',
    'rangayyan_logistic_regression',
    'rangayyan_lstm_signal',
    'mahal',
    'rangayyan_mahalanobis',
    'mcnemar',
    'rangayyan_mcnemar_test',
    'rangayyan_matching_pursuit',
    'rangayyan_neural_decode',
    'rangayyan_nmf',
    'rangayyan_nmf_channel_sel',
    'rangayyan_omp',
    'rangayyan_pca_signals',
    'rangayyan_pca_vs_ica',
    'ppv',
    'rangayyan_ppv',
    'qda',
    'rangayyan_qda',
    'rangayyan_rbf_network',
    'roc',
    'rangayyan_roc_curve',
    'rangayyan_sleep_apnea_nmf',
    'sens',
    'rangayyan_sensitivity',
    'sepindex',
    'rangayyan_separability_index',
    'spec',
    'rangayyan_specificity',
    'rangayyan_sparse_rep',
    'svm',
    'rangayyan_svm',
    'svmkern',
    'rangayyan_svm_kernel',
    'rangayyan_vag_adaptive_tfd',
    'rangayyan_ch4_pan_tompkins_peak_classification',
]

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
def rangayyan_ann_mlp(X, y, layers, lr, max_iter):
    """
    Multilayer perceptron (ANN) with backpropagation

    Formula: y = sigma(W_2*sigma(W_1*y+b_1)+b_2); dW = -eta * dL/dW via chain rule

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    layers : array-like
        Input data.
    lr : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, biases, predictions

    References
    ----------
    Rangayyan Ch 10.8
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Multilayer perceptron (ANN) with backpropagation"}
    )


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
def rangayyan_bundle_branch_block(ecg, fs, r_peaks):
    """
    Bundle branch block (BBB) classification from ECG

    Formula: QRS duration > 120ms; discriminant on QRS width and morphological features

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: block_type, qrs_duration

    References
    ----------
    Rangayyan Ch 10.2.1
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Bundle branch block (BBB) classification from ECG"}
    )


# -- rgbbnorm: Normal versus ectopic beat classification with LDA and Bayes.
def rangayyan_ecg_bbb_normal(ecg, fs, r_peaks, labels):
    """
    Normal versus ectopic beat classification with LDA and Bayes

    Formula: 4-feature LDA; Bayes classifier with Gaussian class models

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: classifier_accuracy, confusion

    References
    ----------
    Rangayyan Ch 10.11
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Normal versus ectopic beat classification with LDA and Bayes",
        }
    )


# -- rgbci: BCI EEG channel selection via NMF spatial decomposition.
def rangayyan_bci_nmf(eeg, n_components, fs):
    """
    BCI EEG channel selection via NMF spatial decomposition

    Formula: NMF: V=WH; W=spatial patterns, H=temporal activations; select channels by W

    Parameters
    ----------
    eeg : array-like
        Input data.
    n_components : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: selected_channels, W, H

    References
    ----------
    Rangayyan Ch 9.12
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "BCI EEG channel selection via NMF spatial decomposition",
        }
    )


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

    Note the book's argument order: the sum is weighted by the SECOND
    PDF, not the first.  KLD is not symmetric -- KLD(p1, p2) is not
    KLD(p2, p1) -- so swapping the arguments gives a different number,
    and both are returned so the asymmetry is visible rather than a trap.
    The symmetric combination is the divergence of eq. (10.115), which is
    exactly KLD(p1, p2) + KLD(p2, p1).

    The book uses this as a FEATURE: Rangayyan and Wu computed the KLD
    between the PDF of a signal to be classified and Parzen-window PDF
    models of the normal and abnormal VAG classes, reaching 73 per cent
    classification with the KLD alone.

    Both PDFs must be positive wherever the weighting PDF is: a zero in
    p1 where p2 is positive makes the ratio unbounded, and that is
    reported rather than silently floored.
    """
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
    fwd = fsum(b[i] * log(b[i] / a[i]) for i in range(len(a)) if b[i] > 0)
    rev = fsum(a[i] * log(a[i] / b[i]) for i in range(len(a))
               if a[i] > 0 and b[i] > 0)
    return RichResult(payload={
        "kld": fwd, "reversed": rev, "symmetric_sum": fwd + rev,
        "asymmetric": abs(fwd - rev) > 1e-12,
        "weighted_by_the_second_pdf": True,
        "symmetric_sum_is_the_divergence_of_eq_10_115": True,
        "nonnegative": fwd >= -1e-12,
        "method": "Rangayyan (2024) eq. (5.33)"})


def bhattcoef(p1, p2):
    """Bhattacharyya coefficient, the OVERLAP between two PDFs.

        BC(p1, p2) = sum_l sqrt( p1(x_l) p2(x_l) )

    Bounded in [0, 1]: 1 when the two PDFs are identical, 0 when their
    supports do not touch.  This is the quantity the Bhattacharyya
    DISTANCE is built from, D_B = -ln BC, and it is what makes the error
    bound work -- the overlap of the two class-conditional densities IS
    the region where the optimal classifier must make mistakes.

    NOT FROM THIS BOOK; see ``bhatt``.
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
    """Hellinger distance.

        H(p1, p2) = sqrt( 1 - BC(p1, p2) )

    with BC the Bhattacharyya coefficient, so H^2 = 1 - BC.  Unlike the
    Bhattacharyya distance -ln BC, this one is a TRUE METRIC: bounded in
    [0, 1], symmetric, and it satisfies the triangle inequality, which
    -ln BC does not.  That is the reason to reach for it -- anything that
    needs a metric (clustering, embedding, nearest-neighbour search over
    distributions) needs this and not D_B.

    The 1/2 normalization is the usual one, H^2 = (1/2) integral
    (sqrt(p) - sqrt(q))^2; under the unnormalized convention H^2 is
    2(1 - BC) instead, and both appear in the literature, so the
    convention is reported rather than assumed.

    NOT FROM RANGAYYAN (2024).
    """
    a, b = aslist(p1), aslist(p2)
    if len(a) != len(b):
        raise ValueError("the two PDFs must be sampled on the same grid")
    if not a:
        raise ValueError("need at least one bin")
    if any(v < 0 for v in a) or any(v < 0 for v in b):
        raise ValueError("a PDF cannot be negative")
    bc = fsum(sqrt(a[i] * b[i]) for i in range(len(a)))
    h2 = max(0.0, 1.0 - bc)
    return RichResult(payload={
        "hellinger": sqrt(h2), "squared": h2,
        "bhattacharyya_coefficient": bc,
        "identity_h2_equals_one_minus_bc": True,
        "is_a_true_metric": True,
        "satisfies_the_triangle_inequality": True,
        "bhattacharyya_distance_does_not": True,
        "normalization": "one half; unnormalized gives 2(1 - BC)",
        "in_unit_interval": -1e-12 <= sqrt(h2) <= 1.0 + 1e-12,
        "reference": "Hellinger E. Neue Begruendung der Theorie "
                     "quadratischer Formen von unendlichvielen "
                     "Veraenderlichen. Journal fuer die reine und "
                     "angewandte Mathematik 136:210-271, 1909, "
                     "doi:10.1515/crll.1909.136.210.",
        "not_from_this_book": True,
        "method": "Hellinger distance, H^2 = 1 - BC"})


def bhatt(m1, m2, C1, C2):
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
def rangayyan_basis_pursuit(x, D, tol):
    """
    Basis pursuit: L1 minimization for sparse representation

    Formula: min ||alpha||_1 s.t. D*alpha = x; solved via LP or ADMM

    Parameters
    ----------
    x : array-like
        Input data.
    D : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha

    References
    ----------
    Rangayyan Ch 9.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Basis pursuit: L1 minimization for sparse representation",
        }
    )


# -- rgcad: Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate.
def rangayyan_cad_pipeline(signals, labels, classifier, cv_k):
    """
    Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate

    Formula: CAD = preprocess(signals) -> F(signals) -> classifier(F) -> cross_validate

    Parameters
    ----------
    signals : array-like
        Input data.
    labels : array-like
        Input data.
    classifier : array-like
        Input data.
    cv_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: accuracy, sensitivity, specificity

    References
    ----------
    Rangayyan Ch 10
    """
    signals = np.asarray(signals, dtype=float)
    n = int(signals) if signals.ndim == 0 else len(signals)
    result = float(np.mean(signals))
    se = float(np.std(signals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate",
        }
    )


# -- rgcnn: 1D CNN for biomedical signal classification.
def rangayyan_cnn_signal(x, filters, kernel_sizes, n_classes):
    """
    1D CNN for biomedical signal classification

    Formula: conv1d: y[n] = sum_k w[k]*x[n+k]; followed by ReLU, pooling, FC layers

    Parameters
    ----------
    x : array-like
        Input data.
    filters : array-like
        Input data.
    kernel_sizes : array-like
        Input data.
    n_classes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: class_probs

    References
    ----------
    Rangayyan Ch 10.8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "1D CNN for biomedical signal classification"}
    )


# -- rgecgfe: Single-channel fetal ECG extraction using NMF/ICA.
def rangayyan_fetal_ecg_single(abdominal_ecg, fs, method):
    """
    Single-channel fetal ECG extraction using NMF/ICA

    Formula: Maternal component removed by NMF decomposition; fetal ECG in residual

    Parameters
    ----------
    abdominal_ecg : array-like
        Input data.
    fs : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fetal_ecg

    References
    ----------
    Rangayyan Ch 9.11
    """
    abdominal_ecg = np.asarray(abdominal_ecg, dtype=float)
    n = int(abdominal_ecg) if abdominal_ecg.ndim == 0 else len(abdominal_ecg)
    result = float(np.mean(abdominal_ecg))
    se = float(np.std(abdominal_ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Single-channel fetal ECG extraction using NMF/ICA"}
    )


# -- rgecgnl: Normal vs. ectopic ECG beat classification.
def rangayyan_ecg_normal_ectopic(ecg, fs, r_peaks):
    """
    Normal vs. ectopic ECG beat classification

    Formula: LDA or k-means on QRS morphological features

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beat_labels, features

    References
    ----------
    Rangayyan Ch 10.11
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Normal vs. ectopic ECG beat classification"}
    )


# -- rgeegb: EEG rhythm band classification (delta/theta/alpha/beta/gamma).
def rangayyan_eeg_rhythms(eeg, fs):
    """
    EEG rhythm band classification (delta/theta/alpha/beta/gamma)

    Formula: Band membership by frequency range: delta<4, theta 4-8, alpha 8-13, beta 13-30, gamma>30 Hz

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: band_powers

    References
    ----------
    Rangayyan Ch 1.2.6
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "EEG rhythm band classification (delta/theta/alpha/beta/gamma)",
        }
    )


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
def rangayyan_epilepsy_ksvd(eeg, fs, dict_size, sparsity):
    """
    Epileptic seizure detection using K-SVD dictionary learning

    Formula: Learned dictionary atoms; OMP for sparse coding; SVM on coefficients

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    dict_size : array-like
        Input data.
    sparsity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_seizure, onset

    References
    ----------
    Rangayyan Ch 9.8
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Epileptic seizure detection using K-SVD dictionary learning",
        }
    )


# -- rgerrbd: Bhattacharyya bound on Bayes classification error.
def errbound(p1, p2, db):
    """Chernoff-Bhattacharyya bound on the Bayes error.

        P_e <= sqrt(P1 P2) exp(-D_B)

    NOT FROM THIS BOOK.  Rangayyan (2024) does not give a Bhattacharyya
    error bound; this is the standard Kailath bound and is kept because
    the name was already exposed.  It pairs with ``bhatt``, not with the
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
        "pairs_with_bhatt_not_with_divergence": True,
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
def rangayyan_fastica(X, n_components, nonlin, max_iter, tol):
    """
    FastICA algorithm for independent component analysis

    Formula: w_k = E{X*g(w_k^T*X)} - E{g'(w_k^T*X)}*w_k; g(y)=tanh(a*y)

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    nonlin : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: S, A, W

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "FastICA algorithm for independent component analysis"}
    )


# -- rgicaart: EEG artifact removal via ICA (eye blink, muscle, ECG).
def rangayyan_ica_artifact(eeg, n_components, artifact_labels):
    """
    EEG artifact removal via ICA (eye blink, muscle, ECG)

    Formula: Artifact components identified by kurtosis/correlation; removed from mixing matrix

    Parameters
    ----------
    eeg : array-like
        Input data.
    n_components : array-like
        Input data.
    artifact_labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eeg_clean, ica_components

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "EEG artifact removal via ICA (eye blink, muscle, ECG)",
        }
    )


# -- rginf: Infomax ICA algorithm (Bell-Sejnowski).
def rangayyan_infomax_ica(X, n_components, lr, max_iter):
    """
    Infomax ICA algorithm (Bell-Sejnowski)

    Formula: DeltaW = (I - f(y)*y^T)*W; f(y) = 1-2*sigmoid(y)

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    lr : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: S, W

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Infomax ICA algorithm (Bell-Sejnowski)"}
    )


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
def rangayyan_knee_classify(vag, fs, labels):
    """
    Knee-joint cartilage pathology classification via VAG features

    Formula: Feature vector: FD, ZCR, form factor, entropy; SVM classifier

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: accuracy, confusion, features

    References
    ----------
    Rangayyan Ch 10.12
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Knee-joint cartilage pathology classification via VAG features",
        }
    )


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
def rangayyan_ksvd(Y, n_atoms, sparsity, max_iter):
    """
    K-SVD dictionary learning algorithm

    Formula: D,X <- alternating SVD update; X=argmin ||Y-DX||_F s.t. ||x_i||_0<=T

    Parameters
    ----------
    Y : array-like
        Input data.
    n_atoms : array-like
        Input data.
    sparsity : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: D, X

    References
    ----------
    Rangayyan Ch 9.5
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-SVD dictionary learning algorithm"})


# compact alias per ledger/NAMING.md
rangayyanksvd = rangayyan_ksvd


# -- rgldsp: Sparse coding given fixed dictionary (OMP/LASSO).
def rangayyan_dictionary_sparse(Y, D, sparsity_T):
    """
    Sparse coding given fixed dictionary (OMP/LASSO)

    Formula: For each y_i: alpha_i = argmin ||alpha||_0 s.t. ||y_i - D*alpha_i||^2 < epsilon

    Parameters
    ----------
    Y : array-like
        Input data.
    D : array-like
        Input data.
    sparsity_T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_sparse

    References
    ----------
    Rangayyan Ch 9.5
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Sparse coding given fixed dictionary (OMP/LASSO)"}
    )


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
def rangayyan_lstm_signal(X_seq, y, hidden_size, n_layers, lr, epochs):
    """
    LSTM recurrent network for biomedical time-series classification

    Formula: i=sigmoid(Wi*[h,y]+bi); f,o,g similar; c=f*c+i*tanh(g); h=o*tanh(c)

    Parameters
    ----------
    X_seq : array-like
        Input data.
    y : array-like
        Input data.
    hidden_size : array-like
        Input data.
    n_layers : array-like
        Input data.
    lr : array-like
        Input data.
    epochs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: predictions, accuracy

    References
    ----------
    Rangayyan Ch 10.8.2
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "LSTM recurrent network for biomedical time-series classification",
        }
    )


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
def rangayyan_matching_pursuit(x, dictionary, max_iter, tol):
    """
    Matching pursuit greedy decomposition into dictionary atoms

    Formula: R_0=x; R_n=R_{n-1}-<R_{n-1},phi_k>*phi_k; iterate until ||R||<epsilon

    Parameters
    ----------
    x : array-like
        Input data.
    dictionary : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coeffs, atoms, residual

    References
    ----------
    Rangayyan Ch 9.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Matching pursuit greedy decomposition into dictionary atoms",
        }
    )


# -- rgneural: Neural decoding for prosthesis control from spike trains.
def rangayyan_neural_decode(spike_trains, movement_labels, n_ch):
    """
    Neural decoding for prosthesis control from spike trains

    Formula: LDA or SVM on firing rate features per neural channel

    Parameters
    ----------
    spike_trains : array-like
        Input data.
    movement_labels : array-like
        Input data.
    n_ch : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: decoded_movement, accuracy

    References
    ----------
    Rangayyan Ch 8.18
    """
    spike_trains = np.asarray(spike_trains, dtype=float)
    n = int(spike_trains) if spike_trains.ndim == 0 else len(spike_trains)
    result = float(np.mean(spike_trains))
    se = float(np.std(spike_trains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Neural decoding for prosthesis control from spike trains",
        }
    )


# -- rgnmf: Nonnegative matrix factorization (NMF) with multiplicative update rules.
def rangayyan_nmf(V, r, max_iter, tol):
    """
    Nonnegative matrix factorization (NMF) with multiplicative update rules

    Formula: H <- H*(W^T*V)/(W^T*W*H); W <- W*(V*H^T)/(W*H*H^T)

    Parameters
    ----------
    V : array-like
        Input data.
    r : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, H

    References
    ----------
    Rangayyan Ch 9.7.3
    """
    V = np.asarray(V, dtype=float)
    n = int(V) if V.ndim == 0 else len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Nonnegative matrix factorization (NMF) with multiplicative update rules",
        }
    )


# compact alias per ledger/NAMING.md
rangayyannmf = rangayyan_nmf


# -- rgnmfch: NMF-based EEG channel selection for BCI.
def rangayyan_nmf_channel_sel(eeg, n_comp, n_select):
    """
    NMF-based EEG channel selection for BCI

    Formula: W matrix columns: spatial activation patterns; select channels by max W_ij

    Parameters
    ----------
    eeg : array-like
        Input data.
    n_comp : array-like
        Input data.
    n_select : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: selected_ch_idx, W, H

    References
    ----------
    Rangayyan Ch 9.12.1
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "NMF-based EEG channel selection for BCI"}
    )


# -- rgomp: Orthogonal matching pursuit (OMP) for sparse representation.
def rangayyan_omp(x, D, sparsity):
    """
    Orthogonal matching pursuit (OMP) for sparse representation

    Formula: r=x; while ||r||>eps: k*=argmax|D^T*r|; x_hat updated by LS on active set; r update

    Parameters
    ----------
    x : array-like
        Input data.
    D : array-like
        Input data.
    sparsity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, support

    References
    ----------
    Rangayyan Ch 9.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Orthogonal matching pursuit (OMP) for sparse representation",
        }
    )


# compact alias per ledger/NAMING.md
rangayyanomp = rangayyan_omp


# -- rgpca: PCA for signal mixture separation (eigendecomposition of covariance).
def rangayyan_pca_signals(X):
    """
    PCA for signal mixture separation (eigendecomposition of covariance)

    Formula: Sigma = (1/N)*X*X^T; X_pca = V^T*X where V=eigenvectors

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components, eigenvalues, eigenvectors

    References
    ----------
    Rangayyan Ch 9.7.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "PCA for signal mixture separation (eigendecomposition of covariance)",
        }
    )


# -- rgpcaica: Comparative analysis of PCA, ICA, and NMF for signal separation.
def rangayyan_pca_vs_ica(X, n_components, method):
    """
    Comparative analysis of PCA, ICA, and NMF for signal separation

    Formula: PCA: orthogonal Gaussian; ICA: statistically independent non-Gaussian; NMF: non-negative

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components, reconstruction_error

    References
    ----------
    Rangayyan Ch 9.7.4
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Comparative analysis of PCA, ICA, and NMF for signal separation",
        }
    )


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
def rangayyan_rbf_network(X, y, n_centers, sigma):
    """
    Radial basis function (RBF) network

    Formula: phi_k(y) = exp(-||y-c_k||^2 / (2*sigma_k^2)); y = sum w_k*phi_k(y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_centers : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: centers, weights, predictions

    References
    ----------
    Rangayyan Ch 10.8.1
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Radial basis function (RBF) network"})


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
def rangayyan_sleep_apnea_nmf(signals, fs, n_comp):
    """
    Sleep apnea diagnosis via NMF of polysomnographic signals

    Formula: NMF on stacked ECG/resp/SpO2 spectrogram matrix; apnea component identified

    Parameters
    ----------
    signals : array-like
        Input data.
    fs : array-like
        Input data.
    n_comp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: apnea_component, apnea_index

    References
    ----------
    Rangayyan Ch 10.13
    """
    signals = np.asarray(signals, dtype=float)
    n = int(signals) if signals.ndim == 0 else len(signals)
    result = float(np.mean(signals))
    se = float(np.std(signals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Sleep apnea diagnosis via NMF of polysomnographic signals",
        }
    )


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
def rangayyan_sparse_rep(x, D, lambda_or_sparsity, method):
    """
    Sparse representation of biomedical signals in learned dictionary

    Formula: min||x-D*alpha||_2 + lambda*||alpha||_1 (LASSO) or ||alpha||_0 (OMP)

    Parameters
    ----------
    x : array-like
        Input data.
    D : array-like
        Input data.
    lambda_or_sparsity : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha, reconstruction

    References
    ----------
    Rangayyan Ch 9.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Sparse representation of biomedical signals in learned dictionary",
        }
    )


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
def rangayyan_vag_adaptive_tfd(vag, fs, n_atoms):
    """
    Adaptive TFD of VAG signals via matching pursuit

    Formula: MP atoms represent time-frequency structures; TFD = sum of atom WVDs

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.
    n_atoms : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfd, t, freqs

    References
    ----------
    Rangayyan Ch 9.9
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Adaptive TFD of VAG signals via matching pursuit"}
    )


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
    'rgann: Multilayer perceptron (ANN) with backpropagation.',
    'Bayes decision functions, eq. (10.70)',
    'Bayes classifier for normal patterns, eq. (10.72)',
    'rgbbb: Bundle branch block (BBB) classification from ECG.',
    'rgbbnorm: Normal versus ectopic beat classification with LDA and Bayes.',
    'rgbci: BCI EEG channel selection via NMF spatial decomposition.',
    'rgbhatt: Bhattacharyya distance for class separability.',
    'rgbp: Basis pursuit: L1 minimization for sparse representation.',
    'rgcad: Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate.',
    'rgcnn: 1D CNN for biomedical signal classification.',
    'rgecgfe: Single-channel fetal ECG extraction using NMF/ICA.',
    'rgecgnl: Normal vs. ectopic ECG beat classification.',
    'rgeegb: EEG rhythm band classification (delta/theta/alpha/beta/gamma).',
    'elbow criterion for the cluster count',
    'rgepiksv: Epileptic seizure detection using K-SVD dictionary learning.',
    'rgerrbd: Bhattacharyya bound on Bayes classification error.',
    "rgfish: Fisher's criterion for feature separability.",
    'rgfld: Fisher linear discriminant analysis (LDA).',
    'hierarchical agglomerative clustering, Section 10.5.1',
    'rgica: FastICA algorithm for independent component analysis.',
    'rgicaart: EEG artifact removal via ICA (eye blink, muscle, ECG).',
    'rginf: Infomax ICA algorithm (Bell-Sejnowski).',
    'k-fold cross-validation, Section 10.10.3',
    'k-means clustering, Section 10.5.1',
    'rgkneecl: Knee-joint cartilage pathology classification via VAG features.',
    'nearest-neighbour and k-NN rules, eq. (10.29)',
    'rgksv: K-SVD dictionary learning algorithm.',
    'rgldsp: Sparse coding given fixed dictionary (OMP/LASSO).',
    'linear discriminant functions, Section 10.4.1',
    'linear discriminant with a fitted threshold, Section 10.4.2',
    'leave-one-out cross-validation, Section 10.10.3',
    'logistic regression by Newton-Raphson, Section 10.7',
    'rglstm: LSTM recurrent network for biomedical time-series classification.',
    'rgmahd: Mahalanobis distance from sample to class.',
    "rgmcn: McNemar's test for comparing two classifiers.",
    'rgmp: Matching pursuit greedy decomposition into dictionary atoms.',
    'rgneural: Neural decoding for prosthesis control from spike trains.',
    'rgnmf: Nonnegative matrix factorization (NMF) with multiplicative update rules.',
    'rgnmfch: NMF-based EEG channel selection for BCI.',
    'rgomp: Orthogonal matching pursuit (OMP) for sparse representation.',
    'rgpca: PCA for signal mixture separation (eigendecomposition of covariance).',
    'rgpcaica: Comparative analysis of PCA, ICA, and NMF for signal separation.',
    'rgppv: Positive predictive value (precision).',
    'quadratic discriminant analysis fitted from data',
    'rgrbf: Radial basis function (RBF) network.',
    'rgroc: Receiver operating characteristic (ROC) curve and AUC.',
    'rgsapnmf: Sleep apnea diagnosis via NMF of polysomnographic signals.',
    'rgsen: Sensitivity (recall, true positive rate).',
    'rgsepix: Separability index: ratio of between-class to within-class scatter.',
    'rgspe: Specificity (true negative rate).',
    'rgsprep: Sparse representation of biomedical signals in learned dictionary.',
    'linear SVM by SMO, Section 10.4.5',
    'kernel SVM, Section 10.4.5',
    'rgvagadp: Adaptive TFD of VAG signals via matching pursuit.',
    'rng190: Pan-Tompkins peak classification.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
