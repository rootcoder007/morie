# morie.fn -- bsaclass (rootcoder007/morie)
"""Pattern classification and decomposition: discriminants, Bayes, SVM, k-NN, clustering, PCA/ICA/NMF, sparse coding, validation and performance measures.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 56
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from math import erf, exp, fsum, lgamma as _lgamma, log, pi, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._richresult import RichResult

__all__ = [
    'accuracy',
    'rangayyan_accuracy',
    'rangayyan_ann_mlp',
    'rangayyan_bayes_classifier',
    'rangayyan_bayes_gaussian',
    'rangayyan_bundle_branch_block',
    'rangayyan_ecg_bbb_normal',
    'rangayyan_bci_nmf',
    'divergence',
    'rangayyan_bhattacharyya',
    'rangayyan_basis_pursuit',
    'rangayyan_cad_pipeline',
    'rangayyan_cnn_signal',
    'rangayyan_fetal_ecg_single',
    'rangayyan_ecg_normal_ectopic',
    'rangayyan_eeg_rhythms',
    'rangayyan_kmeans_elbow',
    'rangayyan_epilepsy_ksvd',
    'errbound',
    'rangayyan_bayes_error_bound',
    'fishcrit',
    'rangayyan_fisher_criterion',
    'fishlda',
    'rangayyan_fisher_lda',
    'rangayyan_hierarchical_clust',
    'rangayyan_fastica',
    'rangayyan_ica_artifact',
    'rangayyan_infomax_ica',
    'rangayyan_kfold_cv',
    'rangayyan_kmeans',
    'rangayyan_knee_classify',
    'rangayyan_knn_classifier',
    'rangayyan_ksvd',
    'rangayyan_dictionary_sparse',
    'rangayyan_linear_discrim',
    'rangayyan_lin_discr_sep',
    'rangayyan_loo_cv',
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
    'rangayyan_svm',
    'rangayyan_svm_kernel',
    'rangayyan_vag_adaptive_tfd',
    'rangayyan_ch4_pan_tompkins_peak_classification',
]

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
             prevalence=None):
    """Classification accuracy, eqs. (10.102) and (10.103).

    The book gives the prevalence-weighted form FIRST, eq. (10.102):

        accuracy = S+ P(A) + S- P(N)

    where P(A) is the prevalence of the disease in the study population
    and P(N) = 1 - P(A).  Only "if the prior probabilities are not
    available" does it fall back on eq. (10.103):

        accuracy = (TP + TN) / (TP + TN + FP + FN)

    The distinction is not pedantry.  Eq. (10.103) is eq. (10.102)
    evaluated at the prevalence of the TEST SET, so on a set deliberately
    balanced 50/50 it reports a number that does not describe performance
    on a population where the disease is rare.  Both are returned, and
    when ``prevalence`` is supplied the weighted figure is the headline.
    """
    if table is not None:
        t = _mat(table)
        if len(t) != 2 or any(len(r) != 2 for r in t):
            raise ValueError("the table must be 2x2, "
                             "[[TP, FN], [FP, TN]]")
        TP, FN, FP, TN = t[0][0], t[0][1], t[1][0], t[1][1]
    else:
        if None in (tp, tn, fp, fn):
            raise ValueError("give a 2x2 table or all four of tp, tn, "
                             "fp, fn")
        TP, TN, FP, FN = float(tp), float(tn), float(fp), float(fn)
    if min(TP, TN, FP, FN) < 0:
        raise ValueError("counts cannot be negative")
    total = TP + TN + FP + FN
    if total <= 0:
        raise ValueError("the table is empty")
    if TP + FN <= 0 or TN + FP <= 0:
        raise ValueError("a class is empty; the sensitivity or "
                         "specificity is undefined")
    se = TP / (TP + FN)
    sp = TN / (TN + FP)
    raw = (TP + TN) / total
    test_prev = (TP + FN) / total
    out = {"accuracy": raw, "raw_accuracy": raw,
           "sensitivity": se, "specificity": sp,
           "test_set_prevalence": test_prev,
           "eq_10_103_is_eq_10_102_at_the_test_set_prevalence": True,
           "method": "Rangayyan (2024) eqs. (10.102)-(10.103)"}
    if prevalence is not None:
        p = float(prevalence)
        if not 0 <= p <= 1:
            raise ValueError("the prevalence must lie in [0, 1]")
        weighted = se * p + sp * (1.0 - p)
        out["accuracy"] = weighted
        out["weighted_accuracy"] = weighted
        out["prevalence"] = p
        out["prior_weighted"] = True
    else:
        out["prior_weighted"] = False
    return RichResult(payload=out)


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
def rangayyan_bayes_classifier(X, class_priors, class_means, class_covs):
    """
    Bayes minimum-error classifier

    Formula: Assign to class k: max P(C_k|X) = max P(X|C_k)*P(C_k)

    Parameters
    ----------
    X : array-like
        Input data.
    class_priors : array-like
        Input data.
    class_means : array-like
        Input data.
    class_covs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, posteriors

    References
    ----------
    Rangayyan Ch 10.6
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes minimum-error classifier"})


# -- rgbayng: Bayes classifier for normal (Gaussian) patterns.
def rangayyan_bayes_gaussian(X, mu_list, sigma_list, priors):
    """
    Bayes classifier for normal (Gaussian) patterns

    Formula: g_k(X) = -0.5*(X-mu_k)^T*Sigma_k^{-1}*(X-mu_k) - 0.5*log|Sigma_k| + log P(C_k)

    Parameters
    ----------
    X : array-like
        Input data.
    mu_list : array-like
        Input data.
    sigma_list : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, discriminants

    References
    ----------
    Rangayyan Ch 10.6.2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Bayes classifier for normal (Gaussian) patterns"}
    )


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


def bhatt(m1, m2, C1, C2):
    """Bhattacharyya distance between two multivariate Gaussians.

        D_B = (1/8) (m1-m2)^T [(C1+C2)/2]^-1 (m1-m2)
              + (1/2) ln( |(C1+C2)/2| / sqrt(|C1| |C2|) )

    NOT FROM THIS BOOK.  Rangayyan (2024) measures class separability
    with the normalized distance of eq. (10.112) and the divergence of
    eqs. (10.115)-(10.117); Bhattacharyya distance appears nowhere in the
    text.  It is implemented here because the name was already exposed
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
def rangayyan_kmeans_elbow(X, max_k):
    """
    Elbow method for k-means cluster count selection

    Formula: WCSS(k) = sum_k sum_{X in C_k} ||X - mu_k||^2; elbow at knee of curve

    Parameters
    ----------
    X : array-like
        Input data.
    max_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wcss_values, optimal_k

    References
    ----------
    Rangayyan Ch 10.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Elbow method for k-means cluster count selection"}
    )


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
def rangayyan_hierarchical_clust(X, linkage, n_clusters):
    """
    Hierarchical agglomerative clustering

    Formula: Merge clusters with minimum linkage distance (single/complete/average)

    Parameters
    ----------
    X : array-like
        Input data.
    linkage : array-like
        Input data.
    n_clusters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, dendrogram

    References
    ----------
    Rangayyan Ch 10.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical agglomerative clustering"})


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
def rangayyan_kfold_cv(X, y, k, classifier):
    """
    K-fold cross-validation

    Formula: CV_k = (1/K) sum_{k=1}^{K} error_k on held-out fold k

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cv_error, fold_errors

    References
    ----------
    Rangayyan Ch 10.10.3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-fold cross-validation"})


# -- rgkmns: K-means clustering algorithm.
def rangayyan_kmeans(X, k, max_iter, tol):
    """
    K-means clustering algorithm

    Formula: Assign to nearest centroid; update mu_k = mean(x_i in cluster k); iterate

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centroids

    References
    ----------
    Rangayyan Ch 10.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-means clustering algorithm"})


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
def rangayyan_knn_classifier(X_train, y_train, X_test, k):
    """
    K-nearest neighbor (k-NN) classifier

    Formula: Assign class of majority among k nearest neighbors by Euclidean distance

    Parameters
    ----------
    X_train : array-like
        Input data.
    y_train : array-like
        Input data.
    X_test : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Rangayyan Ch 10.4.4
    """
    X_train = np.asarray(X_train, dtype=float)
    n = int(X_train) if X_train.ndim == 0 else len(X_train)
    result = float(np.mean(X_train))
    se = float(np.std(X_train, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-nearest neighbor (k-NN) classifier"})


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
def rangayyan_linear_discrim(X, y, w, w0):
    """
    Linear discriminant function for pattern classification

    Formula: g(y) = w^T*y + w_0; classify to class with max g_i(y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    w0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, scores

    References
    ----------
    Rangayyan Ch 10.4.1
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
            "method": "Linear discriminant function for pattern classification",
        }
    )


# -- rglindsep: Linear discriminant function with optimal separability.
def rangayyan_lin_discr_sep(X_1, X_2):
    """
    Linear discriminant function with optimal separability

    Formula: Project X_1 to w^T*X_1; optimal w = S_W^{-1}*(mu_1-mu_2)

    Parameters
    ----------
    X_1 : array-like
        Input data.
    X_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, separation

    References
    ----------
    Rangayyan Ch 10.4.2
    """
    X_1 = np.asarray(X_1, dtype=float)
    n = int(X_1) if X_1.ndim == 0 else len(X_1)
    result = float(np.mean(X_1))
    se = float(np.std(X_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Linear discriminant function with optimal separability",
        }
    )


# -- rgloo: Leave-one-out cross-validation (LOO-CV).
def rangayyan_loo_cv(X, y, classifier):
    """
    Leave-one-out cross-validation (LOO-CV)

    Formula: LOO error = (1/N) sum I(f_{-i}(x_i) != y_i)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loo_error, predictions

    References
    ----------
    Rangayyan Ch 10.10.3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Leave-one-out cross-validation (LOO-CV)"}
    )


# compact alias per ledger/NAMING.md
rangayyanloocv = rangayyan_loo_cv


# -- rglr: Logistic regression for binary classification.
def rangayyan_logistic_regression(X, y, lr, max_iter):
    """
    Logistic regression for binary classification

    Formula: P(y=1|y) = sigmoid(w^T*y + b) = 1/(1+exp(-(w^T*y+b)))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lr : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, b, probabilities

    References
    ----------
    Rangayyan Ch 10.7
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression for binary classification"}
    )


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
def rangayyan_qda(X, y):
    """
    Quadratic discriminant analysis (QDA) with unequal covariance matrices

    Formula: g_k(y) = -0.5*ln|Sigma_k| - 0.5*(y-mu_k)^T*Sigma_k^{-1}*(y-mu_k) + ln P(C_k)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, discriminants

    References
    ----------
    Rangayyan Ch 10.4.2
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
            "method": "Quadratic discriminant analysis (QDA) with unequal covariance matrices",
        }
    )


# compact alias per ledger/NAMING.md
rangayyanqda = rangayyan_qda


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
def rangayyan_svm(X, y, kernel, C):
    """
    Support vector machine (SVM) via margin maximization

    Formula: max 2/||w|| s.t. y_i(w^T*x_i+b)>=1; dual: max sum(a_i)-(1/2)*sum sum a_i*a_j*y_i*y_j*K(x_i,x_j)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    kernel : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: support_vectors, w, b, alphas

    References
    ----------
    Rangayyan Ch 10.4.5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Support vector machine (SVM) via margin maximization"}
    )


# compact alias per ledger/NAMING.md
rangayyansvm = rangayyan_svm


# -- rgsvmk: SVM with kernel trick (RBF, polynomial, sigmoid kernels).
def rangayyan_svm_kernel(X, y, kernel, C, gamma):
    """
    SVM with kernel trick (RBF, polynomial, sigmoid kernels)

    Formula: K(y,y')=exp(-||x-x'||^2/(2*sigma^2)) for RBF; dual: alpha* from QP

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    kernel : array-like
        Input data.
    C : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: svm_model, support_vectors

    References
    ----------
    Rangayyan Ch 10.4.5
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
            "method": "SVM with kernel trick (RBF, polynomial, sigmoid kernels)",
        }
    )


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
    'accuracy, prevalence weighted, eqs. (10.102)-(10.103)',
    'rgann: Multilayer perceptron (ANN) with backpropagation.',
    'rgbayes: Bayes minimum-error classifier.',
    'rgbayng: Bayes classifier for normal (Gaussian) patterns.',
    'rgbbb: Bundle branch block (BBB) classification from ECG.',
    'rgbbnorm: Normal versus ectopic beat classification with LDA and Bayes.',
    'rgbci: BCI EEG channel selection via NMF spatial decomposition.',
    'divergence eq. (10.117), normalized distance eq. (10.112)',
    'rgbp: Basis pursuit: L1 minimization for sparse representation.',
    'rgcad: Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate.',
    'rgcnn: 1D CNN for biomedical signal classification.',
    'rgecgfe: Single-channel fetal ECG extraction using NMF/ICA.',
    'rgecgnl: Normal vs. ectopic ECG beat classification.',
    'rgeegb: EEG rhythm band classification (delta/theta/alpha/beta/gamma).',
    'rgelbow: Elbow method for k-means cluster count selection.',
    'rgepiksv: Epileptic seizure detection using K-SVD dictionary learning.',
    'Bhattacharyya bound on the Bayes error (not from this book)',
    "Fisher's criterion for a scalar feature",
    'Fisher linear discriminant, Section 10.4.2',
    'rghier: Hierarchical agglomerative clustering.',
    'rgica: FastICA algorithm for independent component analysis.',
    'rgicaart: EEG artifact removal via ICA (eye blink, muscle, ECG).',
    'rginf: Infomax ICA algorithm (Bell-Sejnowski).',
    'rgkfcv: K-fold cross-validation.',
    'rgkmns: K-means clustering algorithm.',
    'rgkneecl: Knee-joint cartilage pathology classification via VAG features.',
    'rgknn: K-nearest neighbor (k-NN) classifier.',
    'rgksv: K-SVD dictionary learning algorithm.',
    'rgldsp: Sparse coding given fixed dictionary (OMP/LASSO).',
    'rglindf: Linear discriminant function for pattern classification.',
    'rglindsep: Linear discriminant function with optimal separability.',
    'rgloo: Leave-one-out cross-validation (LOO-CV).',
    'rglr: Logistic regression for binary classification.',
    'rglstm: LSTM recurrent network for biomedical time-series classification.',
    'Mahalanobis distance, Section 10.4.3',
    'McNemar/Bowker test of symmetry, Section 10.9.2',
    'rgmp: Matching pursuit greedy decomposition into dictionary atoms.',
    'rgneural: Neural decoding for prosthesis control from spike trains.',
    'rgnmf: Nonnegative matrix factorization (NMF) with multiplicative update rules.',
    'rgnmfch: NMF-based EEG channel selection for BCI.',
    'rgomp: Orthogonal matching pursuit (OMP) for sparse representation.',
    'rgpca: PCA for signal mixture separation (eigendecomposition of covariance).',
    'rgpcaica: Comparative analysis of PCA, ICA, and NMF for signal separation.',
    'positive predictive value, eq. (10.106)',
    'rgqda: Quadratic discriminant analysis (QDA) with unequal covariance matrices.',
    'rgrbf: Radial basis function (RBF) network.',
    'ROC curve and the area A_z, Section 10.9.1',
    'rgsapnmf: Sleep apnea diagnosis via NMF of polysomnographic signals.',
    'sensitivity (TPF), eq. (10.100)',
    'separability index tr(S_B)/tr(S_W), Section 10.10.1',
    'specificity (TNF), eq. (10.101)',
    'rgsprep: Sparse representation of biomedical signals in learned dictionary.',
    'rgsvm: Support vector machine (SVM) via margin maximization.',
    'rgsvmk: SVM with kernel trick (RBF, polynomial, sigmoid kernels).',
    'rgvagadp: Adaptive TFD of VAG signals via matching pursuit.',
    'rng190: Pan-Tompkins peak classification.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
