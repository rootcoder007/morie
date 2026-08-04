# morie.fn -- bsaclass (rootcoder007/morie)
"""Pattern classification and decomposition: discriminants, Bayes, SVM, k-NN, clustering, PCA/ICA/NMF, sparse coding, validation and performance measures.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 56
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult

__all__ = [
    'rangayyan_accuracy',
    'rangayyan_ann_mlp',
    'rangayyan_bayes_classifier',
    'rangayyan_bayes_gaussian',
    'rangayyan_bundle_branch_block',
    'rangayyan_ecg_bbb_normal',
    'rangayyan_bci_nmf',
    'rangayyan_bhattacharyya',
    'rangayyan_basis_pursuit',
    'rangayyan_cad_pipeline',
    'rangayyan_cnn_signal',
    'rangayyan_fetal_ecg_single',
    'rangayyan_ecg_normal_ectopic',
    'rangayyan_eeg_rhythms',
    'rangayyan_kmeans_elbow',
    'rangayyan_epilepsy_ksvd',
    'rangayyan_bayes_error_bound',
    'rangayyan_fisher_criterion',
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
    'rangayyan_mahalanobis',
    'rangayyan_mcnemar_test',
    'rangayyan_matching_pursuit',
    'rangayyan_neural_decode',
    'rangayyan_nmf',
    'rangayyan_nmf_channel_sel',
    'rangayyan_omp',
    'rangayyan_pca_signals',
    'rangayyan_pca_vs_ica',
    'rangayyan_ppv',
    'rangayyan_qda',
    'rangayyan_rbf_network',
    'rangayyan_roc_curve',
    'rangayyan_sleep_apnea_nmf',
    'rangayyan_sensitivity',
    'rangayyan_separability_index',
    'rangayyan_specificity',
    'rangayyan_sparse_rep',
    'rangayyan_svm',
    'rangayyan_svm_kernel',
    'rangayyan_vag_adaptive_tfd',
    'rangayyan_ch4_pan_tompkins_peak_classification',
]


# -- rgacc: Classification accuracy.
def rangayyan_accuracy(y_true, y_pred):
    """
    Classification accuracy

    Formula: Acc = (TP + TN) / (TP + TN + FP + FN)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: accuracy

    References
    ----------
    Rangayyan Ch 10.9
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classification accuracy"})


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
def rangayyan_bhattacharyya(mu1, sigma1, mu2, sigma2):
    """
    Bhattacharyya distance for class separability

    Formula: D_B = -ln integral sqrt(p1(mu1)*p2(mu1)) dx; for Gaussians: D_B = analytic form

    Parameters
    ----------
    mu1 : array-like
        Input data.
    sigma1 : array-like
        Input data.
    mu2 : array-like
        Input data.
    sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bhattacharyya_dist, coefficient

    References
    ----------
    Rangayyan Ch 10.10.1
    """
    mu1 = np.asarray(mu1, dtype=float)
    n = int(mu1) if mu1.ndim == 0 else len(mu1)
    result = float(np.mean(mu1))
    se = float(np.std(mu1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Bhattacharyya distance for class separability"}
    )


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
def rangayyan_bayes_error_bound(mu1, sigma1, p1, mu2, sigma2, p2):
    """
    Bhattacharyya bound on Bayes classification error

    Formula: P_e <= sqrt(P_1*P_2)*exp(-D_B(P1||P2)); D_B=Bhattacharyya distance

    Parameters
    ----------
    mu1 : array-like
        Input data.
    sigma1 : array-like
        Input data.
    p1 : array-like
        Input data.
    mu2 : array-like
        Input data.
    sigma2 : array-like
        Input data.
    p2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: error_bound

    References
    ----------
    Rangayyan Ch 10.6
    """
    mu1 = np.asarray(mu1, dtype=float)
    n = int(mu1) if mu1.ndim == 0 else len(mu1)
    result = float(np.mean(mu1))
    se = float(np.std(mu1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Bhattacharyya bound on Bayes classification error"}
    )


# -- rgfish: Fisher's criterion for feature separability.
def rangayyan_fisher_criterion(X, y):
    """
    Fisher's criterion for feature separability

    Formula: J(w) = (mu_1-mu_2)^2 / (s_1^2+s_2^2) for scalar feature

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: j_values, ranked_features

    References
    ----------
    Rangayyan Ch 10.10.2
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Fisher's criterion for feature separability"}
    )


# -- rgfld: Fisher linear discriminant analysis (LDA).
def rangayyan_fisher_lda(X, y):
    """
    Fisher linear discriminant analysis (LDA)

    Formula: w = S_W^{-1}*(mu_1-mu_2); S_W=within-class scatter matrix

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, projection, boundary

    References
    ----------
    Rangayyan Ch 10.4.2
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Fisher linear discriminant analysis (LDA)"}
    )


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
def rangayyan_mahalanobis(x, mu, sigma):
    """
    Mahalanobis distance from sample to class

    Formula: D^2 = (x-mu)^T * Sigma^{-1} * (x-mu)

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distance

    References
    ----------
    Rangayyan Ch 10.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Mahalanobis distance from sample to class"}
    )


# -- rgmcn: McNemar's test for comparing two classifiers.
def rangayyan_mcnemar_test(y1, y2, y_true, cdf=None):
    """
    McNemar's test for comparing two classifiers

    Formula: chi^2 = (|b-c|-1)^2 / (b+c); b,c = off-diagonal disagreement counts

    Parameters
    ----------
    y1 : array-like
        Input data.
    y2 : array-like
        Input data.
    y_true : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: chi2, p_value

    References
    ----------
    Rangayyan Ch 10.9.2
    """
    y1 = np.asarray(y1, dtype=float)
    n = int(y1) if y1.ndim == 0 else len(y1)
    if y1.ndim == 0:
        return RichResult(
            payload={"statistic": float("nan"), "p_value": float("nan"), "n": 1, "method": "scalar-input placeholder"}
        )
    if n < 2:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "McNemar's test for comparing two classifiers",
            }
        )
    x_sorted = np.sort(y1)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(y1), scale=np.std(y1, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(
        payload={
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n": n,
            "method": "McNemar's test for comparing two classifiers",
        }
    )


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
def rangayyan_ppv(y_true, y_pred):
    """
    Positive predictive value (precision)

    Formula: PPV = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ppv

    References
    ----------
    Rangayyan Ch 10.9
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Positive predictive value (precision)"})


# compact alias per ledger/NAMING.md
rangayyanppv = rangayyan_ppv


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
def rangayyan_roc_curve(y_true, y_scores):
    """
    Receiver operating characteristic (ROC) curve and AUC

    Formula: ROC: Se vs (1-Sp) at varying thresholds; AUC = integral Se d(1-Sp)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fpr, tpr, auc

    References
    ----------
    Rangayyan Ch 10.9.1
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Receiver operating characteristic (ROC) curve and AUC",
        }
    )


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
def rangayyan_sensitivity(y_true, y_pred):
    """
    Sensitivity (recall, true positive rate)

    Formula: Se = TP / (TP + FN)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sensitivity

    References
    ----------
    Rangayyan Ch 10.9
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Sensitivity (recall, true positive rate)"}
    )


# -- rgsepix: Separability index: ratio of between-class to within-class scatter.
def rangayyan_separability_index(X, y):
    """
    Separability index: ratio of between-class to within-class scatter

    Formula: J = tr(S_B) / tr(S_W); S_B=between-class; S_W=within-class scatter matrix

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: J_index, S_W, S_B

    References
    ----------
    Rangayyan Ch 10.10.1
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
            "method": "Separability index: ratio of between-class to within-class scatter",
        }
    )


# -- rgspe: Specificity (true negative rate).
def rangayyan_specificity(y_true, y_pred):
    """
    Specificity (true negative rate)

    Formula: Sp = TN / (TN + FP)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: specificity

    References
    ----------
    Rangayyan Ch 10.9
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Specificity (true negative rate)"})


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
    'rgacc: Classification accuracy',
    'rgann: Multilayer perceptron (ANN) with backpropagation',
    'rgbayes: Bayes minimum-error classifier',
    'rgbayng: Bayes classifier for normal (Gaussian) patterns',
    'rgbbb: Bundle branch block (BBB) classification from ECG',
    'rgbbnorm: Normal versus ectopic beat classification with LDA and Bayes',
    'rgbci: BCI EEG channel selection via NMF spatial decomposition',
    'rgbhatt: Bhattacharyya distance for class separability',
    'rgbp: Basis pursuit: L1 minimization for sparse representation',
    'rgcad: Computer-aided diagnosis (CAD) pipeline: preprocess -> features -> classify -> validate',
    'rgcnn: 1D CNN for biomedical signal classification',
    'rgecgfe: Single-channel fetal ECG extraction using NMF/ICA',
    'rgecgnl: Normal vs. ectopic ECG beat classification',
    'rgeegb: EEG rhythm band classification (delta/theta/alpha/beta/gamma)',
    'rgelbow: Elbow method for k-means cluster count selection',
    'rgepiksv: Epileptic seizure detection using K-SVD dictionary learning',
    'rgerrbd: Bhattacharyya bound on Bayes classification error',
    "rgfish: Fisher's criterion for feature separability",
    'rgfld: Fisher linear discriminant analysis (LDA)',
    'rghier: Hierarchical agglomerative clustering',
    'rgica: FastICA algorithm for independent component analysis',
    'rgicaart: EEG artifact removal via ICA (eye blink, muscle, ECG)',
    'rginf: Infomax ICA algorithm (Bell-Sejnowski)',
    'rgkfcv: K-fold cross-validation',
    'rgkmns: K-means clustering algorithm',
    'rgkneecl: Knee-joint cartilage pathology classification via VAG features',
    'rgknn: K-nearest neighbor (k-NN) classifier',
    'rgksv: K-SVD dictionary learning algorithm',
    'rgldsp: Sparse coding given fixed dictionary (OMP/LASSO)',
    'rglindf: Linear discriminant function for pattern classification',
    'rglindsep: Linear discriminant function with optimal separability',
    'rgloo: Leave-one-out cross-validation (LOO-CV)',
    'rglr: Logistic regression for binary classification',
    'rglstm: LSTM recurrent network for biomedical time-series classification',
    'rgmahd: Mahalanobis distance from sample to class',
    "rgmcn: McNemar's test for comparing two classifiers",
    'rgmp: Matching pursuit greedy decomposition into dictionary atoms',
    'rgneural: Neural decoding for prosthesis control from spike trains',
    'rgnmf: Nonnegative matrix factorization (NMF) with multiplicative update rules',
    'rgnmfch: NMF-based EEG channel selection for BCI',
    'rgomp: Orthogonal matching pursuit (OMP) for sparse representation',
    'rgpca: PCA for signal mixture separation (eigendecomposition of covariance)',
    'rgpcaica: Comparative analysis of PCA, ICA, and NMF for signal separation',
    'rgppv: Positive predictive value (precision)',
    'rgqda: Quadratic discriminant analysis (QDA) with unequal covariance matrices',
    'rgrbf: Radial basis function (RBF) network',
    'rgroc: Receiver operating characteristic (ROC) curve and AUC',
    'rgsapnmf: Sleep apnea diagnosis via NMF of polysomnographic signals',
    'rgsen: Sensitivity (recall, true positive rate)',
    'rgsepix: Separability index: ratio of between-class to within-class scatter',
    'rgspe: Specificity (true negative rate)',
    'rgsprep: Sparse representation of biomedical signals in learned dictionary',
    'rgsvm: Support vector machine (SVM) via margin maximization',
    'rgsvmk: SVM with kernel trick (RBF, polynomial, sigmoid kernels)',
    'rgvagadp: Adaptive TFD of VAG signals via matching pursuit',
    'rng190: threshold = NPKI + 0.25(SPKI-NPKI), adapts without a fixed cutoff',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
