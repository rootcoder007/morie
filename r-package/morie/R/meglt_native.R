# Exact matrix completion by nuclear norm minimisation.
# Sources: Candes, E. J. & Recht, B. (2009) "Exact Matrix
# Completion via Convex Optimization", Foundations of
# Computational Mathematics 9(6), 717-772,
# doi:10.1007/s10208-009-9045-5, arXiv:0805.4471 -- the sampling
# bound m >= C n^{1.2} r log n, the 1.25 exponent covering all
# ranks, the nuclear norm of eq. (1.4) as the sum of singular
# values and its use in place of the rank, the connection to
# compressed sensing, and the incoherence conditions with the
# motivating example of a matrix whose singular vectors are
# extremely sparse. Cai, J.-F., Candes, E. J. & Shen, Z. (2010)
# "A Singular Value Thresholding Algorithm for Matrix Completion",
# SIAM Journal on Optimization 20(4), 1956-1982,
# doi:10.1137/080738970, arXiv:0810.3286 -- the iterative
# algorithm implemented here. Fazel, M. (2002) Matrix Rank
# Minimization with Applications, PhD thesis, Stanford University
# -- the nuclear norm as the convex envelope of the rank.
#
# Native implementation mirroring Python morie.fn.meglt exactly:
# same nuclear norm, same coherence (row and column, plus the max),
# same sample-bound with the 1.2 / 1.25 exponent check, same SVT
# with the same step size and the projection onto the observed
# entries, same Frobenius relative error.

.meglt_eps <- 1e-12

#' .meglt_svd
#'
#' A step of the meglt_native implementation. Called by \code{svt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; passed to \code{nrow}.
#' @return A list with \code{u}, \code{d}, \code{vt}.
#' @export
.meglt_svd <- function(A) {
  if (is.list(A)) A <- do.call(rbind, lapply(A, as.numeric))
  # nv = 0 threw the right factor away, so the SVT shrinkage had no V
  # to rebuild X from
  s <- svd(A, nu = nrow(A), nv = ncol(A))
  list(u = s$u, d = s$d, vt = t(s$v))
}

#' nuclear_norm
#'
#' A step of the meglt_native implementation. Called by \code{svt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; passed to \code{as.matrix}.
#' @return A numeric value.
#' @export
nuclear_norm <- function(A) {
  M <- as.matrix(A)
  storage.mode(M) <- "double"
  s <- svd(M, nu = 0, nv = 0)$d
  sum(s)
}

#' coherence
#'
#' A step of the meglt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; passed to \code{as.matrix}.
#' @param rank Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.
#' @return A list with \code{mu_row}, \code{mu_col}, \code{mu}, \code{rank}, \code{note}.
#' @export
coherence <- function(A, rank = NULL) {
  M <- as.matrix(A)
  storage.mode(M) <- "double"
  # nv = 0 discards V and t(NULL) errors; keep the right factor
  s_full <- svd(M, nu = nrow(M), nv = ncol(M))
  U <- s_full$u
  s <- s_full$d
  Vt <- t(s_full$v)
  tol <- max(nrow(M), ncol(M)) * (if (length(s) > 0L) s[1] else 0) * 1e-12
  if (is.null(rank)) r <- sum(s > tol) else r <- as.integer(rank)
  if (r < 1L) stop("meglt: the matrix is numerically zero")
  n1 <- nrow(M)
  n2 <- ncol(M)
  mu_u <- max(colSums(U[, seq_len(r), drop = FALSE]^2))
  mu_v <- max(colSums(Vt[, seq_len(r), drop = FALSE]^2))
  list(mu_row = n1 * mu_u / r, mu_col = n2 * mu_v / r,
       mu = max(n1 * mu_u / r, n2 * mu_v / r), rank = r,
       note = "large mu means concentrated singular vectors, and then sampling reveals nothing")
}

#' sample_bound
#'
#' A step of the meglt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param n Coerced to integer by the body, with \code{as.integer}.
#' @param r Coerced to integer by the body, with \code{as.integer}.
#' @param C Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @param exponent Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1.2}.
#' @return A list with \code{m}, \code{fraction}, \code{n}, \code{r}, \code{exponent}, \code{note}.
#' @export
sample_bound <- function(n, r, C = 1.0, exponent = 1.2) {
  if (!(exponent %in% c(1.2, 1.25)))
    stop(sprintf("meglt: the exponent must be 1.2 (moderate rank) or 1.25 (all ranks), got '%s'",
                 paste0("'", as.character(exponent), "'")))
  nn <- as.integer(n)
  rr <- as.integer(r)
  if (nn < 2L || rr < 1L)
    stop("meglt: need n >= 2 and r >= 1")
  m <- as.numeric(C) * (nn ^ as.numeric(exponent)) * rr * log(nn)
  list(m = m, fraction = m / (nn * nn), n = nn, r = rr,
       exponent = as.numeric(exponent),
       note = "the 1.25 exponent holds for ALL ranks; 1.2 assumes the rank is not too large")
}

#' svt
#'
#' A step of the meglt_native implementation. Called by \code{morie_meglt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param M A matrix; passed to \code{as.matrix}.
#' @param observed A matrix; indexed by row and column.
#' @param tau Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.
#' @param step Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1.9}.
#' @param iters Coerced to integer by the body, with \code{as.integer}. Defaults to \code{200L}.
#' @param tol Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-06}.
#' @return A list with \code{estimate}, \code{X}, \code{residual_history}, \code{final_residual}, \code{tau}, \code{n_observed}, \code{fraction_observed}, \code{nuclear_norm}, \code{method}.
#' @export
svt <- function(M, observed, tau = NULL, step = 1.9, iters = 200L,
                tol = 1e-6) {
  A <- as.matrix(M)
  storage.mode(A) <- "double"
  n1 <- nrow(A)
  n2 <- ncol(A)
  obs <- unique(lapply(seq_len(nrow(observed)), function(k) {
    c(as.integer(observed[k, 1]) + 1L, as.integer(observed[k, 2]) + 1L)
  }))
  if (length(obs) == 0L) stop("meglt: no entries were observed")
  t <- if (!is.null(tau)) as.numeric(tau) else 5.0 * sqrt(n1 * n2)
  Y <- matrix(0.0, n1, n2)
  X <- matrix(0.0, n1, n2)
  hist <- c()
  for (it in seq_len(as.integer(iters))) {
    sv <- .meglt_svd(Y)
    U <- sv$u
    s <- sv$d
    Vt <- sv$vt
    sh <- pmax(0, s - t)
    r <- length(sh)
    if (r > 0L) {
      U1 <- U[, seq_len(r), drop = FALSE]
      V1 <- Vt[seq_len(r), , drop = FALSE]
      X <- U1 %*% diag(sh) %*% V1
    } else {
      X <- matrix(0.0, n1, n2)
    }
    res <- 0.0
    for (p in obs) {
      d <- A[p[1], p[2]] - X[p[1], p[2]]
      res <- res + d * d
      Y[p[1], p[2]] <- Y[p[1], p[2]] + as.numeric(step) * d
    }
    hist <- c(hist, sqrt(res))
    if (hist[length(hist)] < as.numeric(tol)) break
  }
  list(estimate = X, X = X, residual_history = hist,
       final_residual = hist[length(hist)], tau = t,
       n_observed = length(obs),
       fraction_observed = length(obs) / (n1 * n2),
       nuclear_norm = nuclear_norm(X),
       method = paste("singular value thresholding for the nuclear-norm program; Candes & Recht (2009), Cai, Candes & Shen (2010)"))
}

#' relative_error
#'
#' A step of the meglt_native implementation. Called by \code{morie_meglt}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param X Numeric; combined arithmetically in the body.
#' @param M A matrix; passed to \code{as.matrix}.
#' @return A numeric value.
#' @export
relative_error <- function(X, M) {
  A <- as.matrix(M)
  storage.mode(A) <- "double"
  num <- sqrt(sum((X - A)^2))
  den <- sqrt(sum(A^2))
  if (den <= .meglt_eps) stop("meglt: the reference matrix is zero")
  num / den
}

matrixcompletion <- svt
matrix_completion_low_rank <- svt

#' .meglt_cheatsheet
#'
#' A step of the meglt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
.meglt_cheatsheet <- function() {
  paste("meglt: most low-rank matrices are recovered EXACTLY from ",
        "m >= C n^1.2 r log n sampled entries -- 1.25 covers all ",
        "ranks. Rank minimisation is NP-hard, so minimise the ",
        "NUCLEAR NORM (sum of singular values), the rank's convex ",
        "surrogate as l1 is for sparsity. INCOHERENCE is required, ",
        "not decorative: e_1 e_1' is rank 1 and unrecoverable ",
        "because nearly every sampled entry is zero. Solved by ",
        "singular value thresholding.", sep = "")
}

#' morie_meglt
#'
#' A step of the meglt_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param M Passed to \code{svt}.
#' @param observed Passed to \code{svt}.
#' @param tau Passed to \code{svt}.
#' @param step Passed to \code{svt}. Defaults to \code{1.9}.
#' @param iters Passed to \code{svt}. Defaults to \code{200L}.
#' @param tol Passed to \code{svt}. Defaults to \code{1e-06}.
#' @return A list with \code{estimate}, \code{X}, \code{residual_history}, \code{final_residual}, \code{tau}, \code{n_observed}, \code{nuclear_norm}, \code{relative_error}, \code{method}.
#' @export
morie_meglt <- function(M, observed, tau = NULL, step = 1.9, iters = 200L,
                        tol = 1e-6) {
  r <- svt(M, observed, tau, step, iters, tol)
  list(estimate = r$estimate, X = r$X,
       residual_history = r$residual_history,
       final_residual = r$final_residual, tau = r$tau,
       n_observed = r$n_observed, nuclear_norm = r$nuclear_norm,
       relative_error = relative_error(r$X, M),
       method = r$method)
}
