# SPDX-License-Identifier: AGPL-3.0-or-later
#' Principal component compression of a marker matrix.
#'
#' Formula: Q = X'X/(n-1) on scaled columns; W the eigenvectors of Q; PC = X W; keep the first k columns
#'
#' @param X One record per row.
#' @param k Number of components retained; None keeps all.
#'
#' @return List with ``scores``, ``loadings``, ``eigenvalues``, ``prop_var``, ``cum_prop``, ``k``, ``n``, ``p``.
#' @references Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, Sect. 2.8 pp. 63-64.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.  Eigenvectors are sign-fixed so the two language arms agree; a repeated eigenvalue leaves its loadings determined only up to a rotation within the eigenspace and is not a stable quantity in either language.
#' @export
Pcadim <- function(X, k = NULL) {
  out <- morie_pca(X, k = k)
  Xm <- .t1_mat(X)
  .t1_result(scores = out$scores, loadings = out$loadings,
             eigenvalues = out$eigenvalues, prop_var = out$prop_var,
             cum_prop = out$cum_prop, k = out$k,
             n = nrow(Xm), p = ncol(Xm),
             method = "PCA compression, MVSML Sect. 2.8")
}
