# morie.fn -- function file (rootcoder007/morie)
# NeRF: a scene as a continuous 5D function.
#
# Sources (from the Python docstring's References section):
# - Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
#   Ramamoorthi, R. & Ng, R. (2020) "NeRF: Representing Scenes as
#   Neural Radiance Fields for View Synthesis", ECCV 2020, LNCS 12346,
#   405-421, doi:10.1007/978-3-030-58452-8_24, arXiv:2003.08934.
# - Max, N. (1995) "Optical models for direct volume rendering", IEEE
#   TVCG 1(2), 99-108, doi:10.1109/2945.468400.
# - Kerbl, B., Kopanas, G., Leimkuhler, T. & Drettakis, G. (2023) "3D
#   Gaussian Splatting for Real-Time Radiance Field Rendering", ACM
#   TOG 42(4), doi:10.1145/3592433.

#' morie_nrfrad
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param payload See Usage.
#' @return Nothing; this branch always raises.
#' @export
morie_nrfrad <- function(payload) {
  if (!is.list(payload) || is.null(payload$op)) {
    stop("nrfrad: payload must be a list with an 'op' field")
  }
  op <- as.character(payload$op)
  if (op == "positional_encoding") {
    return(positional_encoding(payload$p,
                               L = if (is.null(payload$L)) 10 else payload$L,
                               include_input = if (is.null(payload$include_input)) TRUE else payload$include_input))
  }
  if (op == "volume_render") {
    return(volume_render(payload$sigma, payload$colour, payload$t))
  }
  if (op == "sample_pdf") {
    rng <- if (is.null(payload$rng)) .ghc_rng(0) else payload$rng
    eps <- if (is.null(payload$eps)) 1e-5 else payload$eps
    return(sample_pdf(payload$bins, payload$weights,
                      payload$n_samples, rng, eps))
  }
  if (op == "ray_points") {
    rng <- if (is.null(payload$rng)) NULL else payload$rng
    stratified <- if (is.null(payload$stratified)) TRUE else payload$stratified
    return(ray_points(payload$origin, payload$direction,
                      payload$t_near, payload$t_far,
                      payload$n_samples, rng, stratified))
  }
  if (op == "density_is_view_independent") {
    tol <- if (is.null(payload$tol)) 1e-9 else payload$tol
    return(density_is_view_independent(payload$model,
                                       payload$point,
                                       payload$directions,
                                       tol))
  }
  if (op == "cheatsheet") {
    return(.nrfrad_cheatsheet())
  }
  stop("nrfrad: unknown op")
}

#' .nrfrad_vec
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param x See Usage.
#' @return A vector, from \code{as.numeric}.
#' @export
.nrfrad_vec <- function(x) {
  if (is.null(x)) return(numeric(0))
  if (is.list(x) && !is.null(x$sigma)) return(unlist(x$sigma)) # not used
  as.numeric(unlist(x))
}

# Accept either a numeric vector or a list of numerics, mirroring
# k.vec() which iterates over the input.
#' Accept either a numeric vector or a list of numerics, mirroring
#'
#' k.vec() which iterates over the input.
#'
#' @param p See Usage.
#' @return A vector, from \code{as.numeric}.
#' @export
.as_numeric_vec <- function(p) {
  if (is.null(p)) return(numeric(0))
  if (is.list(p)) {
    out <- c()
    for (q in p) out <- c(out, as.numeric(q))
    return(out)
  }
  as.numeric(p)
}

#' positional_encoding
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param p See Usage.
#' @param L Defaults to \code{10}.
#' @param include_input Defaults to \code{TRUE}.
#' @return The value of \code{out}, as built in the body.
#' @export
positional_encoding <- function(p, L = 10, include_input = TRUE) {
  v <- .as_numeric_vec(p)
  if (as.integer(L) < 1L) {
    stop("nrfrad: L must be at least 1")
  }
  out <- if (isTRUE(include_input)) as.numeric(v) else numeric(0)
  for (j in 0:(as.integer(L) - 1L)) {
    f <- (2.0 ^ j) * pi
    for (q in v) {
      out <- c(out, sin(f * q), cos(f * q))
    }
  }
  out
}

#' ray_points
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param origin See Usage.
#' @param direction See Usage.
#' @param t_near See Usage.
#' @param t_far See Usage.
#' @param n_samples See Usage.
#' @param rng Defaults to \code{NULL}.
#' @param stratified Defaults to \code{TRUE}.
#' @return A list with \code{t}, \code{points}, \code{direction}.
#' @export
ray_points <- function(origin, direction, t_near, t_far, n_samples,
                        rng = NULL, stratified = TRUE) {
  o <- .as_numeric_vec(origin)
  d <- .as_numeric_vec(direction)
  nrm <- sqrt(sum(d * d))
  if (nrm <= 1e-12) {
    stop("nrfrad: the ray direction is zero")
  }
  d <- d / nrm
  n <- as.integer(n_samples)
  if (n < 1L || as.numeric(t_far) <= as.numeric(t_near)) {
    stop("nrfrad: need n >= 1 and t_far > t_near")
  }
  step <- (as.numeric(t_far) - as.numeric(t_near)) / n
  ts <- numeric(n)
  for (i in seq_len(n)) {
    lo <- as.numeric(t_near) + (i - 1L) * step
    u <- if (isTRUE(stratified) && !is.null(rng)) {
      as.numeric(.ghc_unif(rng, 1L))
    } else 0.5
    ts[i] <- lo + u * step
  }
  pts <- matrix(0, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    for (a in 1:3) {
      pts[i, a] <- o[a] + ts[i] * d[a]
    }
  }
  list(t = ts, points = pts, direction = d)
}

#' volume_render
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param sigma See Usage.
#' @param colour See Usage.
#' @param t See Usage.
#' @return A list with \code{colour}, \code{weights}, \code{accumulated_alpha}, \code{transmittance_final}, \code{note}.
#' @export
volume_render <- function(sigma, colour, t) {
  s <- .as_numeric_vec(sigma)
  if (is.matrix(colour)) {
    C <- lapply(seq_len(nrow(colour)), function(i) as.numeric(colour[i, ]))
  } else if (is.list(colour)) {
    C <- lapply(colour, function(r) .as_numeric_vec(r))
  } else {
    C <- list(.as_numeric_vec(colour))
  }
  ts <- .as_numeric_vec(t)
  n <- length(s)
  if (!(length(C) == length(ts) && length(ts) == n)) {
    stop("nrfrad: sigma, colour and t differ in length")
  }
  if (any(s < 0.0)) {
    stop("nrfrad: density cannot be negative")
  }
  if (n >= 2L) {
    deltas <- c(ts[2:n] - ts[1:(n - 1L)], 1e10)
  } else {
    deltas <- 1e10
  }
  T_acc <- 1.0
  acc <- rep(0.0, length(C[[1]]))
  weights <- numeric(n)
  for (i in seq_len(n)) {
    a <- 1.0 - exp(-s[i] * deltas[i])
    w <- T_acc * a
    weights[i] <- w
    for (cc in seq_along(acc)) {
      acc[cc] <- acc[cc] + w * C[[i]][cc]
    }
    T_acc <- T_acc * (1.0 - a)
  }
  list(colour = acc, weights = weights,
       accumulated_alpha = sum(weights),
       transmittance_final = T_acc,
       note = paste("differentiable, which is why only posed IMAGES",
                    "are needed -- no 3D supervision"))
}

#' sample_pdf
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param bins See Usage.
#' @param weights See Usage.
#' @param n_samples See Usage.
#' @param rng See Usage.
#' @param eps Defaults to \code{1e-05}.
#' @return A vector, from \code{sort}.
#' @export
sample_pdf <- function(bins, weights, n_samples, rng, eps = 1e-5) {
  b <- .as_numeric_vec(bins)
  w <- .as_numeric_vec(weights) + as.numeric(eps)
  if (!(length(w) == length(b) - 1L || length(w) == length(b))) {
    stop(sprintf("nrfrad: %d weights do not match %d bins",
                 length(w), length(b)))
  }
  tot <- sum(w)
  pdf <- w / tot
  cdf <- numeric(length(pdf))
  acc <- 0.0
  for (i in seq_along(pdf)) {
    acc <- acc + pdf[i]
    cdf[i] <- acc
  }
  out <- numeric(as.integer(n_samples))
  for (k in seq_len(as.integer(n_samples))) {
    u <- as.numeric(.ghc_unif(rng, 1L))
    i <- 1L
    while (i < length(cdf) && u > cdf[i]) {
      i <- i + 1L
    }
    lo <- b[i]
    hi <- b[min(i + 1L, length(b))]
    out[k] <- lo + (hi - lo) * as.numeric(.ghc_unif(rng, 1L))
  }
  sort(out)
}

#' density_is_view_independent
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @param model See Usage.
#' @param point See Usage.
#' @param directions See Usage.
#' @param tol Defaults to \code{1e-09}.
#' @return A list with \code{sigmas}, \code{max_deviation}, \code{view_independent}, \code{note}.
#' @export
density_is_view_independent <- function(model, point, directions, tol = 1e-9) {
  p <- .as_numeric_vec(point)
  ss <- numeric(length(directions))
  for (idx in seq_along(directions)) {
    d <- .as_numeric_vec(directions[[idx]])
    res <- model(p, d)
    ss[idx] <- as.numeric(res$sigma)
  }
  dev <- max(ss) - min(ss)
  list(sigmas = ss, max_deviation = dev,
       view_independent = dev < as.numeric(tol),
       note = paste("sigma from position alone; direction enters only",
                    "for colour"))
}

#' .nrfrad_cheatsheet
#'
#' Part of the nrfrad_native implementation; see the file header for the
#' source it follows.
#'
#' @return A character value.
#' @export
.nrfrad_cheatsheet <- function() {
  paste("nrfrad: a scene IS a continuous 5D function -- position plus",
        "viewing direction to density and radiance -- stored in a plain",
        "MLP; the weights are the scene. DENSITY must come from position",
        "ALONE (direction only affects colour), or the network fakes",
        "specularity by making geometry appear and vanish with the",
        "camera. Classic volume rendering, and because it is",
        "DIFFERENTIABLE the only input is posed images -- no 3D",
        "supervision. POSITIONAL ENCODING is not optional: a",
        "raw-coordinate MLP is low-frequency biased and renders blurry.",
        "Hierarchical sampling reuses the coarse weights as a PDF.")
}
