# Evaluating a treatment prioritization rule: TOC, RATE, Qini.
# Sources: Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E. &
# Wager, S. (2025) "Evaluating Treatment Prioritization Rules via
# Rank-Weighted Average Treatment Effects", JASA 120(549), 38-51,
# doi:10.1080/01621459.2024.2393466 (Definition 1 prioritization
# rule, Definition 2 TOC, Definition 3 RATE, Remark 1 the exact
# null, Sec. 2.2-2.3 the AIPW-score estimator, Theorem 3 and
# Corollary 5 asymptotic linearity and the half-sample bootstrap);
# Sverdrup, E., Wu, H., Athey, S. & Wager, S. (2025) JCGS 34(3),
# 948-960 (the Qini curve under a cost constraint); and Athey, S.,
# Tibshirani, J. & Wager, S. (2019) Ann. Statist. 47(2), 1148-1178
# (the forest whose CATE estimates are the usual priority score).
#
# Native R arm mirroring the Python arm exactly: the same AIPW
# doubly-robust score, the same TOC = running mean of scores down
# the priority ranking minus the grand mean (so TOC(1) is exactly
# zero by construction), the same Qini / AUTOC / uniform RATE
# weights, and the same half-sample bootstrap with the 1/sqrt(2)
# rescale justified by Corollary 5.

.slvgrf_EPS <- 1e-12
.slvgrf_WEIGHTS <- c("qini", "autoc", "uniform")

.slvgrf_vec <- function(x) as.numeric(as.matrix(x))

.check <- function(scores, priority) {
  g <- .slvgrf_vec(scores); s <- .slvgrf_vec(priority)
  if (length(g) != length(s)) {
    stop("slvgrf: ", length(g), " scores but ", length(s),
         " priority values")
  }
  if (length(g) < 2L) {
    stop("slvgrf: need at least 2 units, got ", length(g))
  }
  list(g = g, s = s)
}

#' Doubly-robust AIPW score
#' @export
aipw_scores <- function(Y, W, mu1, mu0, e) {
  y <- .slvgrf_vec(Y); w <- .slvgrf_vec(W)
  m1 <- .slvgrf_vec(mu1); m0 <- .slvgrf_vec(mu0)
  n <- length(y)
  ev <- if (is.numeric(e) && length(e) == 1L) rep(as.numeric(e), n)
        else .slvgrf_vec(e)
  for (nm in c("W", "mu1", "mu0", "e")) {
    v <- switch(nm, W = w, mu1 = m1, mu0 = m0, e = ev)
    if (length(v) != n) {
      stop("slvgrf: ", nm, " has ", length(v),
           " entries for ", n, " units")
    }
  }
  for (v in w) {
    if (!(v == 0 || v == 1)) {
      stop("slvgrf: W must be 0/1, got ", deparse(v))
    }
  }
  for (v in ev) {
    if (!(v > 0 && v < 1)) {
      stop("slvgrf: the propensity must lie strictly in (0, 1); got ",
           deparse(v), " -- overlap fails")
    }
  }
  sapply(seq_len(n), function(i) {
    m1[i] - m0[i] + w[i] * (y[i] - m1[i]) / ev[i] -
      (1 - w[i]) * (y[i] - m0[i]) / (1 - ev[i])
  })
}

#' TOC at u = j/n for j = 1..n (Definition 2)
#' @export
toc_curve <- function(scores, priority) {
  ch <- .check(scores, priority)
  g <- ch$g; s <- ch$s; n <- length(g)
  ord <- order(-s, seq_len(n))
  ate <- sum(g) / n
  run <- 0; toc <- numeric(n); us <- numeric(n)
  for (j in seq_along(ord)) {
    i <- ord[j]
    run <- run + g[i]
    toc[j] <- run / j - ate
    us[j] <- j / n
  }
  list(u = us, toc = toc, ate = ate, order = ord - 1L, n = n)
}

#' RATE: int alpha(u) TOC(u) du
#' @export
rate <- function(scores, priority, weight = "autoc") {
  if (!(weight %in% .slvgrf_WEIGHTS)) {
    stop("slvgrf: weight must be one of ",
         paste(.slvgrf_WEIGHTS, collapse = ", "), ", got ", deparse(weight))
  }
  c <- toc_curve(scores, priority)
  n <- c$n
  val <- if (weight == "qini") {
    sum(c$u * c$toc) / n
  } else {
    sum(c$toc) / n
  }
  list(estimate = val, weight = weight, curve = c, n = n)
}

#' Area under the TOC (RATE with a flat weight)
#' @export
autoc <- function(scores, priority) {
  rate(scores, priority, weight = "autoc")$estimate
}

#' Qini coefficient (RATE with alpha(u) = u)
#' @export
qini_coefficient <- function(scores, priority) {
  rate(scores, priority, weight = "qini")$estimate
}

#' Qini curve: cumulative gain from treating the top fraction
#' @export
qini_curve <- function(scores, priority, cost = NULL) {
  ch <- .check(scores, priority)
  g <- ch$g; s <- ch$s; n <- length(g)
  ord <- order(-s, seq_len(n))
  if (is.null(cost)) {
    cv <- rep(1, n)
  } else {
    cv <- if (is.numeric(cost) && length(cost) == 1L)
      rep(as.numeric(cost), n) else .slvgrf_vec(cost)
    if (length(cv) != n) {
      stop("slvgrf: ", length(cv), " costs for ", n, " units")
    }
    if (any(cv <= 0)) {
      stop("slvgrf: costs must be positive")
    }
  }
  total <- sum(cv)
  run <- 0; spent <- 0; xs <- numeric(n); ys <- numeric(n)
  for (j in seq_along(ord)) {
    i <- ord[j]
    run <- run + g[i]
    spent <- spent + cv[i]
    xs[j] <- spent / total
    ys[j] <- run / n
  }
  list(spend = xs, gain = ys, ate = sum(g) / n, n = n,
       constrained = !is.null(cost))
}

#' Half-sample bootstrap test of RATE = 0
#' @export
rate_test <- function(scores, priority, weight = "autoc", reps = 500L,
                      seed = 0) {
  ch <- .check(scores, priority)
  g <- ch$g; s <- ch$s; n <- length(g)
  if (n < 8L) {
    stop("slvgrf: the half-sample bootstrap needs at least 8 units, ",
         "got ", n)
  }
  theta <- rate(g, s, weight = weight)$estimate
  e <- .ghc_rng(seed)
  half <- n %/% 2L
  draws <- numeric(reps)
  for (r in seq_len(reps)) {
    idx <- sample.int(n, half)
    draws[r] <- rate(g[idx], s[idx], weight = weight)$estimate
  }
  m <- mean(draws)
  v <- var(draws)
  se <- sqrt(max(v, 0) / 2)
  z <- if (se > .slvgrf_EPS) theta / se else 0
  p <- 2 * (1 - pnorm(abs(z)))
  list(estimate = theta, se = se, z = z, p_value = p,
       weight = weight, reps = as.integer(reps), n = n,
       null = paste("the priority score is independent of the ",
                    "treatment effect (Remark 1), NOT that the ATE ",
                    "is zero"),
       method = paste("RATE with half-sample bootstrap, Yadlowsky ",
                      "et al. (2025) Corollary 5"))
}

# house entry point: the package exports one morie_<module>
morie_slvgrf <- rate
