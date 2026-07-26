#' Sample entropy -- Richman & Moorman (2000); NOT covered by Rangayyan
#'
#' Richman & Moorman (2000) sample entropy.
#'
#' \deqn{\mathrm{SampEn}(m, r) = -\ln(A / B)}{SampEn(m, r) = -ln(A / B)}
#'
#' where `B` is the number of unordered template-vector pairs of length
#' `m` within Chebyshev distance `r` and `A` is the count at length
#' `m+1` (self-matches excluded).
#'
#' @param x Numeric vector.
#' @param m Template length (default 2).
#' @param r Tolerance (default `0.2 * sd(x)`).
#' @return Named list `SampEn`, `A`, `B`, `m`, `r`, `n`.
#' @references Richman, J. S. & Moorman, J. R. (2000). Physiological time-series
#'   analysis using approximate entropy and sample entropy. Am J Physiol Heart
#'   Circ Physiol 278(6):H2039-H2049. doi:10.1152/ajpheart.2000.278.6.H2039
#'
#'   Note: this method is NOT in Rangayyan -- the 2024 edition contains no
#'   occurrence of "sample entropy", "approximate entropy", "Pincus" or
#'   "Richman". Both counts use the SAME N-m templates and self-matches are
#'   excluded; these are the two ways SampEn differs from ApEn.
#' @export
#' @examples
#' set.seed(0)
#' rgsam(rnorm(100), m = 2)$SampEn
rgsam <- function(x, m = 2L, r = NULL) {
  N <- length(x)
  if (is.null(r)) r <- 0.2 * stats::sd(x)
  m <- as.integer(m)
  if (N <= m + 1) stop("Need length(x) > m + 1.")
  ## Richman & Moorman count BOTH the length-m and the length-(m+1) matches
  ## over the SAME N-m template vectors. Using N-mm+1 per call gave B one extra
  ## template that A could not have, so the two counts had different
  ## denominators -- reintroducing exactly the bias SampEn was defined to
  ## remove. Mirrors the Python fix in src/morie/fn/rgsam.py.
  nT <- N - m
  if (nT < 2) stop("Need length(x) > m + 1.")
  matches <- function(mm) {
    M <- matrix(0, nrow = nT, ncol = mm)
    for (i in seq_len(nT)) M[i, ] <- x[i:(i + mm - 1)]
    cnt <- 0L
    for (i in seq_len(nT - 1)) {
      d <- apply(abs(sweep(M[(i + 1):nT, , drop = FALSE], 2, M[i, ])), 1, max)
      cnt <- cnt + sum(d <= r)
    }
    cnt
  }
  B <- matches(m)
  A <- matches(m + 1L)
  sampen <- if (A == 0 || B == 0) Inf else -log(A / B)
  list(SampEn = sampen, A = A, B = B, m = m, r = r, n = N)
}

#' @rdname rgsam
#' @keywords internal
#' @export
morie_rangayyan_sample_entropy <- rgsam
