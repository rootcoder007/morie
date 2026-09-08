# morie.fn -- function file (rootcoder007/morie)
# PATE: private aggregation of teacher ensembles.
#
# Sources: Papernot, N., Abadi, M., Erlingsson, U., Goodfellow, I.,
# & Talwar, K. (2017) "Semi-supervised Knowledge Transfer for Deep
# Learning from Private Training Data", ICLR 2017. Equation 1, the
# (2 gamma, 0)-DP per-query guarantee (Theorem 2), the
# data-independent composition of section 3.2
# (4 T gamma^2 + 2 gamma sqrt(2 T ln(1/delta))), the data-dependent
# analysis of section 3.3 with Lemma 4 and Theorem 3, the moments
# accountant that takes the minimum of Theorem 2 and Theorem 3 per
# query, and the tail-bound conversion
# eps = min_lambda (alpha(lambda) + ln(1/delta))/lambda over integer
# lambdas 1..8.

# The vote histogram n_j(x) for each record. teacher_predicts is a
# list of callables mapping rows to labels (or to probability
# vectors, in which case the argmax is the vote).
#' The vote histogram n_j(x) for each record. teacher_predicts is a
#'
#' list of callables mapping rows to labels (or to probability vectors,
#' in which case the argmax is the vote).
#'
#' @param teacher_predicts Coerced to list by the body, with \code{as.list}.
#' @param rows A vector; its length is taken.
#' @param n_classes Defaults to \code{NULL}.
#' @return The value of \code{split}.
#' @export
teacher_votes <- function(teacher_predicts, rows, n_classes = NULL) {
  teachers <- as.list(teacher_predicts)
  if (length(teachers) == 0L)
    stop("pate: at least one teacher is needed")
  votes <- NULL
  for (predict in teachers) {
    out <- predict(rows)
    labels <- integer(length(out))
    for (i in seq_along(out)) {
      v <- out[[i]]
      if (is.list(v) || length(v) > 1L)
        labels[i] <- which.max(v) - 1L
      else
        labels[i] <- as.integer(v)
    }
    if (is.null(votes)) {
      k <- n_classes
      if (is.null(k)) k <- max(labels) + 1L
      votes <- matrix(0L, nrow = length(rows), ncol = k)
    }
    for (i in seq_along(labels)) {
      lab <- labels[i] + 1L
      if (lab > ncol(votes)) {
        extra <- matrix(0L, nrow = nrow(votes), ncol = lab - ncol(votes))
        votes <- cbind(votes, extra)
      }
      votes[i, lab] <- votes[i, lab] + 1L
    }
  }
  split(votes, row(votes))
}

# Equation 1: argmax_j {n_j + Lap(1/gamma)}. One independent
# Laplace draw per class with scale 1/gamma.
#' Equation 1: argmax_j \{n_j + Lap(1/gamma)\}. One independent
#'
#' Laplace draw per class with scale 1/gamma.
#'
#' @param counts Coerced to numeric by the body, with \code{as.numeric}.
#' @param gamma Numeric; combined arithmetically in the body.
#' @param seed Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @return A numeric value.
#' @export
noisy_argmax <- function(counts, gamma, seed = 0) {
  gamma <- as.numeric(gamma)
  if (gamma <= 0) stop("pate: gamma must be positive")
  e <- .ghc_rng(as.numeric(seed))
  n <- as.numeric(counts)
  u <- .ghc_unif(e, length(n)) - 0.5
  lap <- -sign(u) * log(1 - 2 * abs(u)) / gamma
  which.max(n + lap) - 1L
}

# Section 3.2: 4 T gamma^2 + 2 gamma sqrt(2 T ln(1/delta)). The
# composition of T queries, each (2 gamma, 0)-DP, without looking
# at the votes.
#' Section 3.2: 4 T gamma^2 + 2 gamma sqrt(2 T ln(1/delta)). The
#'
#' composition of T queries, each (2 gamma, 0)-DP, without looking at
#' the votes.
#'
#' @param T Numeric; combined arithmetically in the body.
#' @param gamma Numeric; combined arithmetically in the body.
#' @param delta Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
epsilon_data_independent <- function(T, gamma, delta) {
  T <- as.numeric(T)
  gamma <- as.numeric(gamma)
  delta <- as.numeric(delta)
  if (T < 0 || gamma <= 0)
    stop("pate: need T >= 0 and gamma > 0")
  if (!(delta > 0 && delta < 1))
    stop("pate: delta must lie in (0, 1)")
  4 * T * gamma ^ 2 + 2 * gamma * sqrt(2 * T * log(1 / delta))
}

# Lemma 4: sum_{j != j*} (2 + gamma (n_j* - n_j)) /
# (4 exp(gamma (n_j* - n_j))), j* the plurality winner. The bound
# is a bound, so the raw value is also returned.
#' Lemma 4: sum_\{j != j*\} (2 + gamma (n_j* - n_j)) /
#'
#' (4 exp(gamma (n_j* - n_j))), j* the plurality winner. The bound is a
#' bound, so the raw value is also returned.
#'
#' @param counts Coerced to numeric by the body, with \code{as.numeric}.
#' @param gamma Numeric; combined arithmetically in the body.
#' @return A list with \code{bound}, \code{raw}.
#' @export
lemma4_bound <- function(counts, gamma) {
  gamma <- as.numeric(gamma)
  if (gamma <= 0) stop("pate: gamma must be positive")
  n <- as.numeric(counts)
  if (length(n) == 0L) stop("pate: empty vote vector")
  js <- which.max(n) - 1L
  tot <- 0.0
  for (j in seq_along(n) - 1L) {
    if (j == js) next
    gap <- gamma * (n[js + 1L] - n[j + 1L])
    tot <- tot + (2 + gap) / (4 * exp(gap))
  }
  list(bound = min(tot, 1.0), raw = tot)
}

# Theorem 3's data-dependent moment bound, or NA when its
# condition fails. The accountant must fall back on Theorem 2
# outside that range.
#' Theorem 3\'s data-dependent moment bound, or NA when its
#'
#' condition fails. The accountant must fall back on Theorem 2 outside
#' that range.
#'
#' @param q Numeric; combined arithmetically in the body.
#' @param gamma Numeric; combined arithmetically in the body.
#' @param l Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
theorem3_moment <- function(q, gamma, l) {
  q <- as.numeric(q)
  gamma <- as.numeric(gamma)
  if (gamma <= 0) stop("pate: gamma must be positive")
  if (q < 0) stop("pate: q must be non-negative")
  limit <- (exp(2 * gamma) - 1) / (exp(4 * gamma) - 1)
  if (q >= limit) return(NA_real_)
  if (q == 0) return(0.0)
  ratio <- (1 - q) / (1 - exp(2 * gamma) * q)
  if (ratio <= 0) return(NA_real_)
  log((1 - q) * ratio ^ l + q * exp(2 * gamma * l))
}

# Compose the per-query moment bounds and convert to (eps, delta).
#' Compose the per-query moment bounds and convert to (eps, delta)
#'
#' A step of the pate_native implementation. Called by \code{pate}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param vote_counts A vector; its length is taken.
#' @param gamma Numeric; combined arithmetically in the body.
#' @param delta Numeric; combined arithmetically in the body.
#' @param lambdas Optional; may be \code{NULL}. Coerced to integer by the body, with
#' \code{as.integer}.
#' @param data_dependent A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return A list with \code{epsilon}, \code{lambda}, \code{alpha}, \code{delta},
#' \code{queries}, \code{used}.
#' @export
moments_accountant <- function(vote_counts, gamma, delta,
                               lambdas = NULL,
                               data_dependent = TRUE) {
  gamma <- as.numeric(gamma)
  delta <- as.numeric(delta)
  if (!(delta > 0 && delta < 1))
    stop("pate: delta must lie in (0, 1)")
  lams <- if (is.null(lambdas)) seq_len(8L) else as.integer(lambdas)
  if (length(lams) == 0L || any(lams <= 0L))
    stop("pate: lambdas must be positive")
  alpha <- setNames(rep(0.0, length(lams)), as.character(lams))
  used <- list(data_dependent = 0L, data_independent = 0L)
  for (counts in vote_counts) {
    q <- if (isTRUE(data_dependent))
      lemma4_bound(counts, gamma)$bound else 1.0
    for (l in lams) {
      indep <- 2 * gamma ^ 2 * l * (l + 1L)
      dep <- if (isTRUE(data_dependent))
        theorem3_moment(q, gamma, l) else NA_real_
      if (!is.na(dep) && dep < indep) {
        alpha[[as.character(l)]] <- alpha[[as.character(l)]] + dep
        if (l == lams[1L]) used$data_dependent <- used$data_dependent + 1L
      } else {
        alpha[[as.character(l)]] <- alpha[[as.character(l)]] + indep
        if (l == lams[1L]) used$data_independent <- used$data_independent + 1L
      }
    }
  }
  log_inv_delta <- log(1 / delta)
  best <- NULL
  best_l <- lams[1L]
  for (l in lams) {
    eps <- (alpha[[as.character(l)]] + log_inv_delta) / l
    if (is.null(best) || eps < best) {
      best <- eps
      best_l <- l
    }
  }
  list(epsilon = best, lambda = best_l, alpha = alpha,
       delta = delta, queries = length(vote_counts), used = used)
}

# Label `queries` by noisy teacher aggregation and account for the
# privacy cost.
#' Label `queries` by noisy teacher aggregation and account for the
#'
#' privacy cost.
#'
#' @param teacher_predicts Coerced to list by the body, with \code{as.list}.
#' @param queries Coerced to list by the body, with \code{as.list}.
#' @param gamma Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.05}.
#' @param delta Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-05}.
#' @param n_classes Passed to \code{teacher_votes}.
#' @param student_train_fn Optional; may be \code{NULL}. Passed to \code{is.null}.
#' @param student_features Optional; may be \code{NULL}. Coerced to list by the body,
#' with \code{as.list}.
#' @param seed Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.
#' @param lambdas Passed to \code{moments_accountant}.
#' @return A list with \code{estimate}, \code{labels}, \code{clean_labels}, \code{votes},
#' \code{agreement}, \code{epsilon}, \code{epsilon_accountant},
#' \code{epsilon_data_independent}, \code{accountant}, \code{delta}, \code{gamma},
#' \code{n_teachers}, \code{n_queries}, \code{student}, \code{note}, \code{method}.
#' @export
pate <- function(teacher_predicts, queries, gamma = 0.05, delta = 1e-5,
                 n_classes = NULL, student_train_fn = NULL,
                 student_features = NULL, seed = 0, lambdas = NULL) {
  rows <- as.list(queries)
  if (length(rows) == 0L) stop("pate: no queries to label")
  votes <- teacher_votes(teacher_predicts, rows, n_classes)
  # The Laplace draws in noisy_argmax are independent of the
  # SplitMix64 stream that the rest of the package uses -- this
  # module ships its own RNG to match the Python arm's `rng`.
  e <- .ghc_rng(as.numeric(seed))
  labels <- integer(length(rows))
  for (i in seq_along(votes)) {
    counts <- votes[[i]]
    gamma_ <- as.numeric(gamma)
    if (gamma_ <= 0) stop("pate: gamma must be positive")
    u <- .ghc_unif(e, length(counts)) - 0.5
    lap <- -sign(u) * log(1 - 2 * abs(u)) / gamma_
    labels[i] <- which.max(counts + lap) - 1L
  }
  clean <- integer(length(votes))
  for (i in seq_along(votes))
    clean[i] <- which.max(votes[[i]]) - 1L
  acct <- moments_accountant(votes, as.numeric(gamma),
                             as.numeric(delta), lambdas)
  indep <- epsilon_data_independent(length(rows), as.numeric(gamma),
                                    as.numeric(delta))
  eps <- min(acct$epsilon, indep)
  student <- NULL
  if (!is.null(student_train_fn)) {
    X <- if (is.null(student_features)) rows else as.list(student_features)
    if (length(X) != length(labels))
      stop("pate: student_features must be one per query")
    student <- student_train_fn(X, labels)
  }
  list(estimate = labels, labels = labels, clean_labels = clean,
       votes = votes,
       agreement = sum(labels == clean) / as.numeric(length(labels)),
       epsilon = eps,
       epsilon_accountant = acct$epsilon,
       epsilon_data_independent = indep,
       accountant = acct,
       delta = as.numeric(delta), gamma = as.numeric(gamma),
       n_teachers = length(as.list(teacher_predicts)),
       n_queries = length(rows),
       student = student,
       note = paste("clean_labels are the noiseless plurality and ",
                    "carry NO privacy guarantee; the semi-supervised ",
                    "GAN student of section 4 is not implemented",
                    sep = ""),
       method = paste("PATE noisy teacher aggregation (Papernot et ",
                      "al. 2017)", sep = ""))
}

#' .pate_cheatsheet
#'
#' A step of the pate_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
#' @examples
#' res <- .pate_cheatsheet()
#' res
.pate_cheatsheet <- function() {
  paste("pate: private aggregation of teacher ensembles (Papernot et ",
        "al. 2017). Teachers trained on disjoint partitions vote; the ",
        "student sees only argmax_j {n_j + Lap(1/gamma)} (eq.1), which ",
        "is (2 gamma, 0)-DP per query since one record moves one ",
        "teacher. Two accountings: data-independent ",
        "4 T gamma^2 + 2 gamma sqrt(2 T ln(1/delta)), which reproduces ",
        "the paper's 26 and 5.80; and data-dependent, where a strong ",
        "quorum makes the majority near-certain, q from Lemma 4 feeds ",
        "Theorem 3, the smaller of that and 2 gamma^2 l(l+1) is taken ",
        "per query, summed by Theorem 1, and converted by the tail ",
        "bound eps = min_lambda (alpha + ln(1/delta))/lambda. The ",
        "noiseless plurality is NOT private.", sep = "")
}

morie_pate <- pate
