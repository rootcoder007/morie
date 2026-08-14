# Set Transformer attention pooling (MAB / SAB / PMA).
# Source: Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S. and
# Teh, Y. W. (2019), "Set Transformer: A Framework for Attention-
# based Permutation-Invariant Neural Networks", ICML 2019 (PMLR 97),
# arXiv:1810.00825. Implemented equations: MAB(X, Y) = LN(H + rFF(H))
# with H = LN(X + Multihead(X, Y, Y)) (Eq 7), SAB(X) = MAB(X, X)
# (Eq 8) and PMA_k(Z) = MAB(S, rFF(Z)) (Sec 3.2), with S a learnable
# k x d seed matrix. Because every row of Z enters only through K
# and V of the attention, PMA is permutation invariant in the set
# elements, which is the test anchor.
#
# Native R arm mirroring the Python arm exactly: the same single-head
# scaled dot-product attention with explicit Wq/Wk/Wv projections,
# the same row-wise relu h W1 + b1 followed by W2 + b2 in rFF, and
# the same per-row LayerNorm on the residual sum.

.EPS <- 1e-5

.ln <- function(row, eps = .EPS) {
  n <- length(row)
  mu <- sum(row) / n
  var <- sum((row - mu)^2) / n
  s <- sqrt(var + eps)
  (row - mu) / s
}

.rff <- function(M, W1, b1, W2, b2) {
  H <- M %*% W1
  R <- t(apply(H, 1, function(row)
    pmax(0, row + as.numeric(b1))))
  O <- R %*% W2
  O + matrix(as.numeric(b2), nrow = nrow(O), ncol = ncol(O),
             byrow = TRUE)
}

.attend <- function(X, Y, Wq, Wk, Wv) {
  Q <- X %*% Wq
  K <- Y %*% Wk
  V <- Y %*% Wv
  dk <- ncol(Q)
  Sraw <- (Q %*% t(K)) * (1.0 / sqrt(dk))
  m <- apply(Sraw, 1, max)
  e <- exp(Sraw - m)
  z <- rowSums(e)
  W <- e / z
  O <- W %*% V
  list(O = O, W = W)
}

.mab <- function(X, Y, p) {
  out <- .attend(X, Y, p$Wq, p$Wk, p$Wv)
  A <- out$O; W <- out$W
  H <- t(apply(X + A, 1, .ln))
  F <- .rff(H, p$W1, p$b1, p$W2, p$b2)
  O <- t(apply(H + F, 1, .ln))
  list(O = O, W = W)
}

#' PMA_k attention pooling from Lee et al. (2019)
#' @export
setT <- function(Z, S, params) {
  Za <- as.matrix(Z); storage.mode(Za) <- "double"
  Sa <- as.matrix(S); storage.mode(Sa) <- "double"
  if (ncol(Za) != ncol(Sa)) {
    stop("setT: Z width ", ncol(Za), " != seed width ", ncol(Sa))
  }
  need <- c("Wq", "Wk", "Wv", "W1", "b1", "W2", "b2")
  miss <- setdiff(need, names(params))
  if (length(miss) > 0L) stop("setT: params is missing ",
                              paste(miss, collapse = ", "))
  p <- lapply(params, function(v) {
    m <- as.matrix(v); storage.mode(m) <- "double"; m
  })
  Zl <- Za
  Sl <- Sa
  FZ <- .rff(Zl, p$W1, p$b1, p$W2, p$b2)
  out <- .mab(Sl, FZ, p)
  list(output = out$O, attention = out$W,
       k = nrow(Sl), estimate = as.numeric(out$O[1L, 1L]),
       n = as.integer(nrow(Za)),
       method = paste("Set Transformer PMA_k(Z) = MAB(S, rFF(Z)) ",
                      "(Lee et al. 2019, Eq 7 + Sec 3.2)"))
}

#' Back-compatible wrapper over `setT` (old stub name)
#' @export
set_transformer <- function(X = NULL, k = NULL, S = NULL,
                            params = NULL) {
  if (is.null(X) || is.null(S) || is.null(params)) {
    stop("set_transformer: X, S and params are required")
  }
  setT(X, S, params)
}

# house entry point: the package exports one morie_<module>
morie_setT <- setT
