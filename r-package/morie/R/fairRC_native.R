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

.EPS <- 1e-12

.cutoffs <- function(N, step = 10L) {
  n <- as.integer(N); s <- as.integer(step)
  if (n < s) {
    stop(sprintf("fairRC: the ranking of %d is shorter than the first cut-off %d", n, s))
  }
  seq(s, n, by = s)
}

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
      a <- min(max(p, .EPS), 1 - .EPS)
      b <- min(max(P, .EPS), 1 - .EPS)
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

.normalizer <- function(protected, measure = "rND", step = 10L) {
  n <- length(protected)
  npos <- sum(as.integer(protected))
  worst <- c(rep(0L, n - npos), rep(1L, npos))
  z <- .raw(worst, measure, step)
  if (z > .EPS) z else 1
}

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

rND <- function(protected, step = 10L, normalize = TRUE) {
  .measure(protected, "rND", step, normalize)
}

rKL <- function(protected, step = 10L, normalize = TRUE) {
  .measure(protected, "rKL", step, normalize)
}

rRD <- function(protected, step = 10L, normalize = TRUE) {
  p <- as.integer(as.logical(protected))
  cav <- NULL
  if (sum(p) > 0.5 * length(p)) {
    cav <- "rRD is NOT APPLICABLE here: the protected group is the MAJORITY, and rRD does not treat the two groups symmetrically"
  }
  .measure(p, "rRD", step, normalize, cav)
}

morie_fairRC <- function(protected, step = 10L, normalize = TRUE) {
  rND(protected, step = step, normalize = normalize)
}
