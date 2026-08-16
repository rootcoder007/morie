# morie.fn -- function file (rootcoder007/morie)
# Sources:
#   Schutt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S.,
#   Tkatchenko, A. & Muller, K.-R. (2017) "SchNet: A continuous-filter
#   convolutional neural network for modeling quantum interactions",
#   Advances in Neural Information Processing Systems 30 (NeurIPS
#   2017), 991-1001, arXiv:1706.08566.
#   Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl,
#   G. E. (2017) "Neural Message Passing for Quantum Chemistry",
#   ICML 2017, PMLR 70, 1263-1272, arXiv:1704.01212.

.schN_EPS <- 1e-12

#' gaussian_expansion
#'
#' Part of the schN_native implementation; see the file header for the
#' source it follows.
#'
#' @param r See Usage.
#' @param mu_min Defaults to \code{0}.
#' @param mu_max Defaults to \code{6}.
#' @param n_gaussians Defaults to \code{25}.
#' @param gamma Defaults to \code{NULL}.
#' @return A numeric value.
#' @export
gaussian_expansion <- function(r, mu_min = 0.0, mu_max = 6.0,
                               n_gaussians = 25, gamma = NULL) {
  n <- as.integer(n_gaussians)
  if (n < 2L) stop("schN: at least 2 Gaussians are needed")
  lo <- as.numeric(mu_min); hi <- as.numeric(mu_max)
  if (hi <= lo) stop("schN: mu_max must exceed mu_min")
  step <- (hi - lo) / (n - 1L)
  g <- if (is.null(gamma)) 1.0 / (2.0 * step ^ 2) else as.numeric(gamma)
  mus <- lo + step * seq(0, n - 1L)
  v <- as.numeric(r)
  exp(-g * (v - mus) ^ 2)
}

#' cosine_cutoff
#'
#' Part of the schN_native implementation; see the file header for the
#' source it follows.
#'
#' @param r See Usage.
#' @param cutoff Defaults to \code{5}.
#' @return One of two values, depending on the branch taken.
#' @export
cosine_cutoff <- function(r, cutoff = 5.0) {
  rc <- as.numeric(cutoff)
  if (rc <= 0.0) stop("schN: the cutoff must be positive")
  v <- as.numeric(r)
  if (v < rc) 0.5 * (cos(pi * v / rc) + 1.0) else 0.0
}

#' cfconv
#'
#' Part of the schN_native implementation; see the file header for the
#' source it follows.
#'
#' @param X See Usage.
#' @param R See Usage.
#' @param filter_net See Usage.
#' @param cutoff Defaults to \code{5}.
#' @param ... Passed through.
#' @return The value of \code{out}, as built in the body.
#' @export
cfconv <- function(X, R, filter_net, cutoff = 5.0, ...) {
  feats <- lapply(X, function(r) as.numeric(r))
  pos <- lapply(R, function(r) as.numeric(r))
  n <- length(feats)
  d <- length(feats[[1L]])
  if (length(pos) != n) {
    stop(sprintf("schN: %d feature rows but %d positions", n, length(pos)))
  }
  out <- list()
  for (i in seq_len(n)) {
    acc <- rep(0.0, d)
    for (j in seq_len(n)) {
      if (i == j) next
      diff_ <- pos[[i]] - pos[[j]]
      r <- sqrt(sum(diff_ * diff_))
      w <- as.numeric(filter_net(gaussian_expansion(r, ...)))
      fc <- cosine_cutoff(r, cutoff)
      if (length(w) != d) {
        stop(sprintf("schN: the filter is %d-dimensional but the features are %d",
                     length(w), d))
      }
      acc <- acc + feats[[j]] * w * fc
    }
    out[[length(out) + 1L]] <- acc
  }
  out
}

#' forces_from_energy
#'
#' Part of the schN_native implementation; see the file header for the
#' source it follows.
#'
#' @param energy_fn See Usage.
#' @param R See Usage.
#' @param h Defaults to \code{1e-05}.
#' @return A list with \code{estimate}, \code{forces}, \code{net_force}, \code{method}, \code{note}.
#' @export
forces_from_energy <- function(energy_fn, R, h = 1e-5) {
  pos <- lapply(R, function(r) as.numeric(r))
  n <- length(pos)
  d <- length(pos[[1L]])
  F <- list()
  for (i in seq_len(n)) {
    row <- numeric(d)
    for (a in seq_len(d)) {
      up <- lapply(pos, function(p) as.numeric(p))
      dn <- lapply(pos, function(p) as.numeric(p))
      up[[i]][a] <- up[[i]][a] + h
      dn[[i]][a] <- dn[[i]][a] - h
      row[a] <- -(as.numeric(energy_fn(up)) - as.numeric(energy_fn(dn))) / (2.0 * h)
    }
    F[[length(F) + 1L]] <- row
  }
  nf <- numeric(d)
  for (i in seq_len(n)) for (a in seq_len(d)) nf[a] <- nf[a] + F[[i]][a]
  list(estimate = F, forces = F, net_force = nf,
       method = "forces as the negative gradient of the energy; Schutt et al. (2017)",
       note = "conservative and equivariant by construction; a separate force head would be neither")
}

#' invariance_error
#'
#' Part of the schN_native implementation; see the file header for the
#' source it follows.
#'
#' @param energy_fn See Usage.
#' @param R See Usage.
#' @param Q See Usage.
#' @param g Defaults to \code{NULL}.
#' @return A list with \code{energy_error}, \code{force_error}, \code{energy_invariant}, \code{forces_equivariant}, \code{note}.
#' @export
invariance_error <- function(energy_fn, R, Q, g = NULL) {
  pos <- lapply(R, function(r) as.numeric(r))
  d <- length(pos[[1L]])
  gv <- if (is.null(g)) rep(0.0, d) else as.numeric(g)
  Qm <- as.matrix(Q)
  rot <- lapply(pos, function(p) as.numeric(Qm %*% p) + gv)
  e0 <- as.numeric(energy_fn(pos))
  e1 <- as.numeric(energy_fn(rot))
  F0 <- forces_from_energy(energy_fn, pos)$forces
  F1 <- forces_from_energy(energy_fn, rot)$forces
  want <- lapply(F0, function(f) as.numeric(Qm %*% f))
  fe <- 0
  for (i in seq_along(F1)) {
    for (a in seq_len(d)) {
      v <- abs(F1[[i]][a] - want[[i]][a])
      if (v > fe) fe <- v
    }
  }
  list(energy_error = abs(e1 - e0), force_error = fe,
       energy_invariant = abs(e1 - e0) < 1e-8,
       forces_equivariant = fe < 1e-5,
       note = "energy INVARIANT, forces EQUIVARIANT -- two different properties from one design choice")
}

.schN_cheatsheet <- function() {
  paste("schN: a convolution needs a grid and atoms have none, so",
        "make the filter a FUNCTION of interatomic distance -- a",
        "continuous-filter convolution, generated by a small",
        "network from the distance. Positions enter only as",
        "||r_i - r_j||, so the energy is rotationally INVARIANT;",
        "forces come from -dE/dr, so they are EQUIVARIANT and the",
        "field is conservative, which a separate force head would",
        "not be. Expand the distance in GAUSSIANS or the filter",
        "varies too sharply for molecular dynamics; a cosine",
        "cutoff keeps neighbourhood changes continuous.")
}

#' morie_schN
#'
#' Part of the schN_native implementation; see the file header for the
#' source it follows.
#'
#' @param ... Passed through.
#' @return The value of \code{cfconv}.
#' @export
morie_schN <- function(...) {
  cfconv(...)
}

schnet <- cfconv
