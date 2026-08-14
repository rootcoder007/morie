# Uniform angle quantisation on [-pi, pi). Source: this module is a
# standard flat quantiser on the circle; the MSE bound Delta^2/12 is
# the standard result for a uniform density on an interval, which
# here is the circle with the sectors unwrapped, and the wraparound
# difference is the only thing that distinguishes angles from
# ordinary scalars. The "anchor" claim in the Python module's prose
# is that the empirical MSE approaches Delta^2/12 when the angles are
# uniform on the circle.
#
# Native implementation mirroring Python morie.fn.tqang exactly: the
# same wrap to [-pi, pi), the same sector index
# k = floor((w + pi) / Delta) with the same out-of-range clamp
# (floating-point angles that land exactly at +pi wrap to -pi but
# can still produce k = n_levels), the same midpoint reconstruction
# -pi + (k + 0.5) Delta, and the same wrapped errors so the
# reported error is the true angular distance at the seam.

# --- helpers --------------------------------------------------------

.tqang_two_pi <- 2 * pi

#' Wrap an angle to \code{[-pi, pi)}
#' @param theta Numeric scalar.
#' @return Numeric scalar in \code{[-pi, pi)}.
#' @export
morie_tqang_wrap_angle <- function(theta) {
  t <- (as.numeric(theta) + pi) %% .tqang_two_pi
  if (t < 0) t <- t + .tqang_two_pi
  t - pi
}

#' Signed shortest difference \code{a - b}, in \code{[-pi, pi)}
#' @return Numeric scalar.
#' @export
morie_tqang_angular_difference <- function(a, b) {
  morie_tqang_wrap_angle(as.numeric(a) - as.numeric(b))
}

#' Uniform quantisation of angles to \code{2^bits} equal sectors
#'
#' Returns a list with the sector indices, the reconstructed angles,
#' the wrapped errors, the empirical MSE, the worst-case absolute
#' error, the sector width \code{delta}, \code{half_delta}, the
#' theoretical MSE bound \code{Delta^2/12}, the \code{bits} and
#' \code{levels}.
#' @param theta Numeric vector of angles.
#' @param bits Bits per symbol (1..30).
#' @export
morie_tqang_quantize_angles <- function(theta, bits = 4L) {
  b <- as.integer(bits)
  if (!(b >= 1L && b <= 30L))
    stop("quantize_angles: bits must lie in 1..30")
  n_levels <- bitwShiftL(1L, b)
  delta <- .tqang_two_pi / n_levels

  t_in <- as.numeric(theta)
  if (length(t_in) == 0L) {
    return(list(estimate = numeric(0), indices = integer(0),
                values = numeric(0), errors = numeric(0),
                mse = 0, max_abs_error = 0, delta = delta,
                half_delta = delta / 2, mse_bound = delta * delta / 12,
                bits = b, levels = n_levels,
                method = "Uniform angle quantisation on [-pi, pi), midpoint reconstruction, wrapped error"))
  }
  w <- vapply(t_in, morie_tqang_wrap_angle, numeric(1))
  k <- floor((w + pi) / delta)
  k[k >= n_levels] <- n_levels - 1L
  k[k < 0L] <- 0L
  rec <- -pi + (k + 0.5) * delta
  err <- vapply(seq_along(w), function(i)
    morie_tqang_angular_difference(w[i], rec[i]), numeric(1))
  mse <- mean(err^2)
  list(estimate = rec, indices = as.integer(k), values = rec,
       errors = err, mse = mse,
       max_abs_error = max(abs(err)),
       delta = delta, half_delta = delta / 2,
       mse_bound = delta * delta / 12, bits = b, levels = n_levels,
       method = "Uniform angle quantisation on [-pi, pi), midpoint reconstruction, wrapped error")
}

#' Compact alias \code{tqang}
#' @export
morie_tqang_tqang <- morie_tqang_quantize_angles

#' Backward-compatible alias for the generated stub
#' @export
morie_tqang_turboquant_angle_quantization <- morie_tqang_quantize_angles

#' One-line rationale mirroring the Python cheatsheet
#' @return Character.
#' @export
morie_tqang_cheatsheet <- function() {
  paste0("tqang: 2^b equal sectors, delta = 2pi/2^b, codeword ",
         "-pi + (k+0.5) delta; |err| <= delta/2, MSE -> delta^2/12; ",
         "all errors use the WRAPPED difference.")
}

# house entry point: the package exports one morie_<module>
morie_tqang <- morie_tqang_wrap_angle
