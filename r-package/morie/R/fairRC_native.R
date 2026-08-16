# morie.fn -- function file (rootcoder007/morie)
# Measuring fairness in ranked outputs.
#
# References
# Yang, K. & Stoyanovich, J. (2017) "Measuring Fairness in Ranked
# Outputs", Proceedings of the 29th International Conference on
# Scientific and Statistical Database Management (SSDBM '17),
# arXiv:1610.08559. Sec. 3.
# Jarvelin, K. & Kekalainen, J. (2002) "Cumulated gain-based evaluation
# of IR techniques", ACM Transactions on Information Systems 20(4),
# 422-446. The logarithmic discount.

.fairRC_EPS <- 1e-12

#' .cutoffs
#'
#' A step of the fairRC_native implementation. Called by \code{.measure}, \code{.raw}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param N Coerced to integer by the body, with \code{as.integer}.
#' @param step Coerced to integer by the body, with \code{as.integer}. Defaults to \code{10L}.
#' @return The value of \code{seq}.
#' @export
.cutoffs <- function(N, step = 10L) {
  n <- as.integer(N); s <- as.integer(step)
  if (n < s) {
    stop(sprintf("fairRC: the ranking of %d is shorter than the first cut-off %d", n, s))
  }
  seq(s, n, by = s)
}

#' .raw
#'
#' A step of the fairRC_native implementation. Called by \code{.measure}, \code{.normalizer}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected A vector; its length is taken and its elements indexed.
#' @param measure One of \code{"rKL"}, \code{"rND"}.
#' @param step Passed to \code{.cutoffs}.
#' @return The value of \code{tot}, as built in the body.
#' @export
.raw <- function(protected, measure, step) {
  N <- length(protected)
  P <- sum(protected) / N
  tot <- 0
  for (i in .cutoffs(N, step)) {
    p <- sum(protected[seq_len(i)]) / i
    w <- 1 / log(i, 2)
    if (measure == "rND") {
      tot <- tot + w * abs(p - P)
    } else if (measure == "rKL") {
      a <- min(max(p, .fairRC_EPS), 1 - .fairRC_EPS)
      b <- min(max(P, .fairRC_EPS), 1 - .fairRC_EPS)
      tot <- tot + w * (a * log(a / b) + (1 - a) * log((1 - a) / (1 - b)))
    } else {
      npos <- sum(protected[seq_len(i)])
      nneg <- i - npos
      r1 <- if (nneg == 0 || npos == 0) 0 else npos / nneg
      NP <- sum(protected); NN <- N - NP
      r2 <- if (NN == 0 || NP == 0) 0 else NP / NN
      tot <- tot + w * abs(r1 - r2)
    }
  }
  tot
}

#' .normalizer
#'
#' A step of the fairRC_native implementation. Called by \code{.measure}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected A vector; its length is taken.
#' @param measure Passed to \code{.raw}. Defaults to \code{"rND"}.
#' @param step Passed to \code{.raw}. Defaults to \code{10L}.
#' @return One of two values, depending on the branch taken.
#' @export
.normalizer <- function(protected, measure = "rND", step = 10L) {
  n <- length(protected)
  npos <- sum(as.integer(protected))
  worst <- c(rep(0L, n - npos), rep(1L, npos))
  z <- .raw(worst, measure, step)
  if (z > .fairRC_EPS) z else 1
}

#' .measure
#'
#' A step of the fairRC_native implementation. Called by \code{rKL}, \code{rND}, \code{rRD}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected Coerced to logical by the body, with \code{as.logical}.
#' @param measure One of \code{"rKL"}, \code{"rND"}, \code{"rRD"}.
#' @param step Passed to \code{.raw}.
#' @param normalize A flag; the body branches on it.
#' @param caveat Defaults to \code{NULL}.
#' @return The value of \code{pay}, as built in the body.
#' @export
.measure <- function(protected, measure, step, normalize, caveat = NULL) {
  if (!(measure %in% c("rND", "rKL", "rRD"))) {
    stop(sprintf("fairRC: measure must be one of %s, got %s",
                 paste(c("rND", "rKL", "rRD"), collapse = ", "),
                 sQuote(measure)))
  }
  p <- as.integer(as.logical(protected))
  if (length(p) == 0L) stop("fairRC: the ranking is empty")
  s <- sum(p)
  if (s == 0L || s == length(p)) {
    stop("fairRC: fairness is undefined when every item is in one group")
  }
  raw <- .raw(p, measure, step)
  z <- if (isTRUE(normalize)) .normalizer(p, measure, step) else 1
  pay <- list(
    estimate = raw / z,
    value = raw / z,
    raw = raw,
    normalizer = z,
    measure = measure,
    protected_share = s / length(p),
    cutoffs = .cutoffs(length(p), step),
    method = "Yang & Stoyanovich (2017) Sec. 3",
    note = "0 is fairest; the best value is reached when the top-i share matches the POPULATION share, not 50/50"
  )
  if (!is.null(caveat)) pay$caveat <- caveat
  pay
}

#' rND
#'
#' A step of the fairRC_native implementation. Called by \code{morie_fairRC}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected Passed to \code{.measure}.
#' @param step Passed to \code{.measure}. Defaults to \code{10L}.
#' @param normalize Passed to \code{.measure}. Defaults to \code{TRUE}.
#' @return The value of \code{.measure}.
#' @export
rND <- function(protected, step = 10L, normalize = TRUE) {
  .measure(protected, "rND", step, normalize)
}

#' rKL
#'
#' A step of the fairRC_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected Passed to \code{.measure}.
#' @param step Passed to \code{.measure}. Defaults to \code{10L}.
#' @param normalize Passed to \code{.measure}. Defaults to \code{TRUE}.
#' @return The value of \code{.measure}.
#' @export
rKL <- function(protected, step = 10L, normalize = TRUE) {
  .measure(protected, "rKL", step, normalize)
}

#' rRD
#'
#' A step of the fairRC_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected Coerced to logical by the body, with \code{as.logical}.
#' @param step Passed to \code{.measure}. Defaults to \code{10L}.
#' @param normalize Passed to \code{.measure}. Defaults to \code{TRUE}.
#' @return The value of \code{.measure}.
#' @export
rRD <- function(protected, step = 10L, normalize = TRUE) {
  p <- as.integer(as.logical(protected))
  cav <- NULL
  if (sum(p) > 0.5 * length(p)) {
    cav <- "rRD is NOT APPLICABLE here: the protected group is the MAJORITY, and rRD does not treat the two groups symmetrically"
  }
  .measure(p, "rRD", step, normalize, cav)
}

#' morie_fairRC
#'
#' A step of the fairRC_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param protected See Usage.
#' @param step Defaults to \code{10L}.
#' @param normalize Defaults to \code{TRUE}.
#' @return The value of \code{rND}.
#' @export
morie_fairRC <- function(protected, step = 10L, normalize = TRUE) {
  rND(protected, step = step, normalize = normalize)
}
