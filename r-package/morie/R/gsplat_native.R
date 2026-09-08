# 3D Gaussian splatting: an explicit primitive that rasterises.
# Sources: Kerbl, B., Kopanas, G., Leimkuhler, T. and Drettakis, G.
# (2023) "3D Gaussian Splatting for Real-Time Radiance Field
# Rendering", ACM Transactions on Graphics 42(4), Article 139,
# doi:10.1145/3592433, arXiv:2308.04079 (3D Gaussians, covariance
# factorisation, adaptive density control, fast rasterisation);
# Zwicker, M. et al. (2001) "EWA volume splatting", Proceedings
# Visualization 2001, 29-36, doi:10.1109/VISUAL.2001.964490 (the EWA
# splat); Mildenhall, B. et al. (2020) "NeRF", ECCV 2020, LNCS 12346,
# 405-421, doi:10.1007/978-3-030-58452-8_24 (the implicit alternative).
#
# Native implementation mirroring Python morie.fn.gsplat exactly: the
# same scale-rotation factorisation of the covariance, the same
# EWA projection, the same front-to-back alpha compositing and the
# same adaptive density control.

.GSPLAT_EPS <- 1e-12

#' Build the 3D covariance as Sigma = R S S' R'
#'
#' Every reachable parameter value is positive semi-definite by
#' construction; gradient descent on the six raw entries would not
#' stay there.
#'
#' @param scale Length-3 positive scale vector.
#' @param quaternion Length-4 (unnormalised) rotation quaternion.
#' @return A list with \code{covariance}, \code{rotation},
#'   \code{scale} and \code{note}.
#' @references Kerbl, B. et al. (2023).
#' @export
morie_gsplat_covariance <- function(scale, quaternion) {
  s <- as.numeric(scale)
  if (length(s) != 3L || any(s <= 0))
    stop("gsplat: scales must be three positive numbers")
  v <- as.numeric(quaternion)
  n <- sqrt(sum(v * v))
  if (n <= .GSPLAT_EPS)
    stop("gsplat: the rotation quaternion is zero")
  w <- v[1L] / n
  x <- v[2L] / n
  y <- v[3L] / n
  z <- v[4L] / n
  R <- matrix(c(1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x),
                1 - 2 * (x * x + y * y)),
              nrow = 3L, ncol = 3L, byrow = TRUE)
  M <- R * matrix(s, nrow = 3L, ncol = 3L, byrow = TRUE)
  S <- tcrossprod(M)
  list(covariance = S, rotation = R, scale = s,
       note = paste0("PSD by construction, which raw entries would ",
                     "not be"))
}

#' Positive semi-definiteness check
#'
#' Computes the eigenvalues of a symmetric matrix and tests the
#' smallest against a tolerance.
#'
#' @param S 3x3 symmetric matrix.
#' @param tol Tolerance for \code{min_eigenvalue >= tol}.
#' @return A list with \code{eigenvalues}, \code{min_eigenvalue} and
#'   \code{psd}.
#' @references Kerbl, B. et al. (2023).
#' @export
morie_gsplat_psd <- function(S, tol = -1e-9) {
  M <- apply(S, c(1L, 2L), as.numeric)
  ev <- eigen(M, symmetric = TRUE, only.values = TRUE)$values
  list(eigenvalues = ev, min_eigenvalue = min(ev),
       psd = min(ev) >= as.numeric(tol))
}

#' EWA projection of the 3D covariance
#'
#' \eqn{\Sigma' = J W \Sigma W^\top J^\top}, the affine approximation
#' to perspective projection that keeps the splat closed-form and
#' therefore fast.
#'
#' @param S 3x3 covariance.
#' @param W 3x3 world-to-camera transform.
#' @param J 2x3 Jacobian of the perspective projection.
#' @return A list with \code{projected} (2x2) and \code{dim}.
#' @references Zwicker, M. et al. (2001); Kerbl, B. et al. (2023).
#' @export
morie_gsplat_project <- function(S, W, J) {
  C <- apply(S, c(1L, 2L), as.numeric)
  Wm <- apply(W, c(1L, 2L), as.numeric)
  Jm <- apply(J, c(1L, 2L), as.numeric)
  T <- Jm %*% Wm
  out <- T %*% C %*% t(T)
  list(projected = out, dim = nrow(out),
       note = paste0("an affine approximation to perspective, hence ",
                     "closed form and fast"))
}

#' Front-to-back alpha compositing
#'
#' Same image formation as volume rendering, which is why the two
#' representations are interchangeable at the pixel.
#'
#' @param colours Matrix of per-Gaussian colours (n x 3 or n x d).
#' @param alphas Numeric vector of opacities in \[0, 1\].
#' @param depths Optional depth per Gaussian; sorts back-to-front.
#' @return A list with \code{colour}, \code{transmittance},
#'   \code{coverage} and \code{note}.
#' @references Kerbl, B. et al. (2023); Mildenhall, B. et al. (2020).
#' @export
morie_gsplat_composite <- function(colours, alphas, depths = NULL) {
  C <- apply(colours, c(1L, 2L), as.numeric)
  a <- as.numeric(alphas)
  if (nrow(C) != length(a))
    stop(paste0("gsplat: ", nrow(C), " colours but ", length(a),
                " alphas"))
  if (any(a < 0) || any(a > 1))
    stop("gsplat: alphas must lie in [0,1]")
  if (is.null(depths)) {
    order <- seq_len(nrow(C))
  } else {
    order <- order(as.numeric(depths))
  }
  T_ <- 1.0
  acc <- rep(0.0, ncol(C))
  for (i in order) {
    acc <- acc + T_ * a[i] * C[i, ]
    T_ <- T_ * (1.0 - a[i])
  }
  list(colour = acc, transmittance = T_, coverage = 1.0 - T_,
       note = paste0("identical compositing to volume rendering; ",
                     "only the primitive and traversal differ"))
}

#' Adaptive density control
#'
#' Large positional gradient with a small Gaussian means
#' under-reconstruction (clone); with a large Gaussian it means
#' over-reconstruction (split). Near-transparent Gaussians are
#' pruned. The Gaussian count is not fixed in advance.
#'
#' @param gradients Per-Gaussian positional gradient magnitudes.
#' @param scales Per-Gaussian scale (e.g. max eigenvalue).
#' @param opacities Per-Gaussian opacity.
#' @param grad_threshold Clone / split threshold on the gradient.
#' @param scale_threshold Split threshold on the scale.
#' @param opacity_threshold Prune threshold on the opacity.
#' @return A list with \code{clone}, \code{split}, \code{prune},
#'   \code{n_before} and \code{n_after}.
#' @references Kerbl, B. et al. (2023).
#' @export
morie_gsplat_density <- function(gradients, scales, opacities,
                                 grad_threshold = 2e-4,
                                 scale_threshold = 0.01,
                                 opacity_threshold = 0.005) {
  g <- as.numeric(gradients)
  s <- as.numeric(scales)
  o <- as.numeric(opacities)
  if (length(g) != length(s) || length(g) != length(o))
    stop("gsplat: the inputs differ in length")
  clone <- integer(0)
  split <- integer(0)
  prune <- integer(0)
  for (i in seq_along(g)) {
    if (o[i] < as.numeric(opacity_threshold)) {
      prune <- c(prune, i)
    } else if (g[i] > as.numeric(grad_threshold)) {
      if (s[i] > as.numeric(scale_threshold)) split <- c(split, i)
      else clone <- c(clone, i)
    }
  }
  list(estimate = list(clone = clone, split = split, prune = prune),
       clone = clone, split = split, prune = prune,
       n_before = length(g),
       n_after = length(g) + length(clone) + length(split) - length(prune),
       method = "adaptive density control; Kerbl et al. (2023)",
       note = paste0("under-reconstruction clones, over-reconstruction ",
                     "splits, transparent prunes"))
}

# house entry point: the package exports one morie_<module>
morie_gsplat <- morie_gsplat_covariance
