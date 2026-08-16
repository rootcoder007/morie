# morie.fn -- function file (rootcoder007/morie)
# E(n)-equivariant graph convolution -- re-export of egnnL.
# egcn and egnnL are two ledger rows citing the same paper
# (Satorras, Hoogeboom & Welling 2021). They are kept as one
# implementation with a re-export so the two entries cannot drift
# apart, exactly as timesf re-exports timesfm.
#
# References
# Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n) Equivariant
# Graph Neural Networks", Proceedings of the 38th International
# Conference on Machine Learning (ICML 2021), PMLR 139, 9323-9332,
# arXiv:2102.09844. Sec. 3 (the EGCL of eqs. (3)-(6), with C = 1/(M-1)).
# Sec. 3.1 (the equivariance condition Qx + g). Sec. 3.2 (the momentum
# variant replacing eq. (4)).
# Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K. &
# Riley, P. (2018) "Tensor Field Networks", arXiv:1802.08219.

# Re-export from egnnL
#' Re-export from egnnL
#'
#' Part of the egcn_native implementation; see the file header for the
#' source it follows.
#'
#' @param H See Usage.
#' @param X See Usage.
#' @param layers See Usage.
#' @param phi_e See Usage.
#' @param phi_x See Usage.
#' @param phi_h See Usage.
#' @param A Defaults to \code{NULL}.
#' @param C Defaults to \code{NULL}.
#' @return The value of \code{morie_egnnL}.
#' @export
morie_egcn <- function(H, X, layers, phi_e, phi_x, phi_h, A = NULL, C = NULL) {
  morie_egnnL(H = H, X = X, layers = layers, phi_e = phi_e,
              phi_x = phi_x, phi_h = phi_h, A = A, C = C)
}
