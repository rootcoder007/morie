# SPDX-License-Identifier: AGPL-3.0-or-later
#' Baujat plot coordinates
#'
#' x_i = w_i (y_i - mu_F)^2, the contribution to Cochran's Q, and
#' y_i = (mu_F - mu_F(-i))^2 / Var(mu_F(-i)), the influence on the pooled
#' effect, with fixed-effect weights.  Source consulted: Baujat, Mahe, Pignon
#' and Hill (2002), Statistics in Medicine 21, 2641-2652.
#'
#' @param yi,vi study effects and their within-study variances.
#' @return list: estimate, x, y, pooled, Q, n, method.
#' @keywords internal
#' @examples
#' mabaujat(c(0.1, 0.3, -0.2, 0.45), c(0.02, 0.05, 0.03, 0.08))$x
#' @export
mabaujat <- function(yi, vi) {
  y <- as.numeric(yi); v <- as.numeric(vi); k <- length(y)
  fe <- k02fe(y, v)
  w <- 1 / v
  xs <- w * (y - fe$mu)^2
  ys <- numeric(k)
  for (i in seq_len(k)) {
    d <- k02fe(y[-i], v[-i])
    ys[i] <- (fe$mu - d$mu)^2 / d$var
  }
  list(estimate = max(ys), x = xs, y = ys, pooled = fe$mu, Q = fe$Q, n = k,
       method = "Baujat plot coordinates (Baujat, Mahe, Pignon & Hill 2002)")
}

# CANONICAL TEST
# r <- mabaujat(c(0.10,0.30,-0.20,0.45,0.05,0.22), c(0.02,0.05,0.03,0.08,0.01,0.04))
# stopifnot(abs(sum(r$x) - r$Q) < 1e-12)

#' @rdname mabaujat
#' @keywords internal
#' @export
morie_mabaujat <- mabaujat
