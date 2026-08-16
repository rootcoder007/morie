# Reversible-jump MCMC: sampling across models of differing dimension.
# Sources: Green, P. J. (1995) "Reversible jump Markov chain Monte Carlo
# computation and Bayesian model determination", *Biometrika* 82(4),
# 711-732, doi:10.1093/biomet/82.4.711, for the §3-2 general theory
# (the acceptance ratio of eq. 7, the symmetric-measure dimension
# matching Assumption), §3-3 the switching template with bijective
# (theta, u) tuples of equal total length, and §4 the change-point
# application: Equation 9 (the step-function log-likelihood), §4-3
# (move probabilities and the birth/death acceptance ratio), §4-4 (the
# reported run, the defaults here). Peskun, P. H. (1973) "Optimum
# Monte-Carlo sampling using Markov chains", *Biometrika* 60(3),
# 607-612, for the optimality of the acceptance probability. The
# within-model fixed-dimension update is a symmetric Gaussian random
# walk, the default for reversible jump.

.JACOBIAN_ROUTES <- c("analytic", "numeric")
.MOVE_KEYS <- c("frm", "to", "n_u", "n_u_rev", "propose", "transform")


#' .unif_stream
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}, \code{reversible_jump_mcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param seed Passed to \code{.ghc_rng}.
#' @param block Coerced to integer by the body, with \code{as.integer}. Defaults to \code{8192L}.
#' @return The value of \code{function}.
#' @export
.unif_stream <- function(seed, block = 8192L) {
  e <- new.env(parent = emptyenv())
  e$buf <- numeric(0)
  e$i <- 1L
  e$stream <- 0L
  e$block <- as.integer(block)

  e$uni <- function() {
    if (e$i > length(e$buf)) {
      u <- .ghc_unif(e$rng, e$block)
      e$buf <- u
      e$i <- 1L
    }
    v <- e$buf[[e$i]]
    e$i <- e$i + 1L
    if (v <= 0.0) v <- 1e-15
    else if (v >= 1.0) v <- 1.0 - 1e-15
    v
  }
  e$rng <- .ghc_rng(seed)
  function() e$uni()
}

#' .logabsdet
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{numeric_log_jacobian}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param a A matrix; passed to \code{nrow}.
#' @return The value of \code{total}, as built in the body.
#' @export
.logabsdet <- function(a) {
  n <- nrow(a)
  if (n == 0) return(0.0)
  m <- a
  total <- 0.0
  for (col in seq_len(n)) {
    piv <- col
    best <- abs(m[col, col])
    if (col < n) {
      for (r in (col + 1L):n) {
        if (abs(m[r, col]) > best) { piv <- r; best <- abs(m[r, col]) }
      }
    }
    if (best < 1e-300) return(-Inf)
    if (piv != col) {
      tmp <- m[col, ]; m[col, ] <- m[piv, ]; m[piv, ] <- tmp
    }
    total <- total + log(abs(m[col, col]))
    if (col < n) {
      for (r in (col + 1L):n) {
        f <- m[r, col] / m[col, col]
        if (f == 0.0) next
        m[r, col:n] <- m[r, col:n] - f * m[col, col:n]
      }
    }
  }
  total
}

#' numeric_log_jacobian
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{reversible_jump_mcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param mapfun Accepted by the signature and not used anywhere in the body.
#' @param z A vector; its length is taken and its elements indexed.
#' @param h Numeric; combined arithmetically in the body. Defaults to \code{1e-06}.
#' @return The value of \code{.logabsdet}.
#' @export
numeric_log_jacobian <- function(mapfun, z, h = 1e-6) {
  z <- as.numeric(z)
  n <- length(z)
  out0 <- mapfun(z)
  if (length(out0) != n)
    stop(sprintf("bayrjmcmc: the bijection maps %d values to %d; dimension matching requires n1 + m1 == n2 + m2",
                 n, length(out0)))
  if (n == 0) return(0.0)
  jac <- matrix(0.0, n, n)
  for (j in seq_len(n)) {
    step <- h * max(1.0, abs(z[j]))
    zp <- z; zm <- z
    zp[j] <- zp[j] + step
    zm[j] <- zm[j] - step
    fp <- mapfun(zp); fm <- mapfun(zm)
    jac[, j] <- (fp - fm) / (2.0 * step)
  }
  .logabsdet(jac)
}

#' check_dimension_matching
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{reversible_jump_mcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param models A vector; its length is taken and its elements indexed.
#' @param moves See Usage.
#' @return The value of \code{by_pair}, as built in the body.
#' @export
check_dimension_matching <- function(models, moves) {
  if (length(models) == 0)
    stop("bayrjmcmc: no models given")
  for (nm in names(models)) {
    spec <- models[[nm]]
    if (is.null(spec$dim) || is.null(spec$logpost))
      stop(sprintf("bayrjmcmc: model %r needs 'dim' and 'logpost'", nm))
    if (as.integer(spec$dim) < 0)
      stop(sprintf("bayrjmcmc: model %r has negative dim", nm))
  }
  by_pair <- list()
  for (mv in moves) {
    for (key in .MOVE_KEYS)
      if (is.null(mv[[key]]))
        stop(sprintf("bayrjmcmc: a move is missing %r", key))
    if (is.null(models[[mv$frm]]) || is.null(models[[mv$to]]))
      stop(sprintf("bayrjmcmc: move %r -> %r names an unknown model", mv$frm, mv$to))
    if (mv$frm == mv$to)
      stop(sprintf("bayrjmcmc: %r -> %r is a within-model move; give it as 'within', not as a jump", mv$frm, mv$to))
    pair <- paste(mv$frm, mv$to, sep = "|")
    if (!is.null(by_pair[[pair]]))
      stop(sprintf("bayrjmcmc: two moves given for %r -> %r", mv$frm, mv$to))
    by_pair[[pair]] <- mv
  }
  for (mv in moves) {
    rev_key <- paste(mv$to, mv$frm, sep = "|")
    rev <- by_pair[[rev_key]]
    if (is.null(rev))
      stop(sprintf("bayrjmcmc: move %r -> %r has no reverse move; detailed balance is imposed within each move type, so the reverse must be supplied", mv$frm, mv$to))
    n1 <- as.integer(models[[mv$frm]]$dim)
    n2 <- as.integer(models[[mv$to]]$dim)
    m1 <- as.integer(mv$n_u)
    m2 <- as.integer(mv$n_u_rev)
    if (m1 < 0 || m2 < 0)
      stop(sprintf("bayrjmcmc: move %r -> %r has negative n_u", mv$frm, mv$to))
    if (n1 + m1 != n2 + m2)
      stop(sprintf("bayrjmcmc: move %r -> %r violates dimension matching: n1 + m1 = %d + %d != %d + %d = n2 + m2",
                   mv$frm, mv$to, n1, m1, n2, m2))
    if (as.integer(rev$n_u) != m2 || as.integer(rev$n_u_rev) != m1)
      stop(sprintf("bayrjmcmc: move %r -> %r declares u of length %d and a reverse u of length %d, but the reverse move declares %d and %d",
                   mv$frm, mv$to, m1, m2, as.integer(rev$n_u), as.integer(rev$n_u_rev)))
  }
  by_pair
}

#' rj_log_acceptance
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{reversible_jump_mcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param logpost_from Numeric; combined arithmetically in the body.
#' @param logpost_to Numeric; combined arithmetically in the body.
#' @param log_j_from Numeric; combined arithmetically in the body.
#' @param log_j_to Numeric; combined arithmetically in the body.
#' @param logq_u Numeric; combined arithmetically in the body.
#' @param logq_u_rev Numeric; combined arithmetically in the body.
#' @param log_jacobian Numeric; combined arithmetically in the body.
#' @return The value of \code{(}.
#' @export
rj_log_acceptance <- function(logpost_from, logpost_to, log_j_from,
                              log_j_to, logq_u, logq_u_rev, log_jacobian) {
  ((logpost_to - logpost_from)
   + (log_j_to - log_j_from)
   + (logq_u_rev - logq_u)
   + log_jacobian)
}

#' .rw_within
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{reversible_jump_mcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param theta A vector; its length is taken and its elements indexed.
#' @param uni Accepted by the signature and not used anywhere in the body.
#' @param scale Numeric; combined arithmetically in the body.
#' @return The value of \code{out}, as built in the body.
#' @export
.rw_within <- function(theta, uni, scale) {
  out <- numeric(length(theta))
  for (i in seq_along(theta)) {
    u1 <- uni(); u2 <- uni()
    z <- sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
    out[i] <- theta[i] + scale * z
  }
  out
}

#' reversible_jump_mcmc
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{morie_bayrjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param models A vector; indexed elementwise.
#' @param moves Passed to \code{check_dimension_matching}.
#' @param init_model Passed to \code{sprintf}.
#' @param init_theta Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{numeric(0)}.
#' @param n_iter A count; the body uses it as \code{seq_len(...)}. Defaults to \code{10000}.
#' @param burn_in Numeric; combined arithmetically in the body. Defaults to \code{0}.
#' @param thin Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param seed Passed to \code{.unif_stream}. Defaults to \code{0}.
#' @param within Optional; may be \code{NULL}. A vector; indexed elementwise.
#' @param within_scale Passed to \code{.rw_within}. Defaults to \code{0.5}.
#' @param within_weight Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.
#' @param move_weight Passed to \code{\%||\%}. Defaults to \code{1}.
#' @param jacobian One of \code{"analytic"}, \code{"numeric"}. Defaults to \code{"analytic"}.
#' @param keep_chain A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return A list with \code{model_freq}, \code{visits}, \code{n_kept}, \code{accept}, \code{tried}, \code{chain}, \code{jacobian}, \code{method}, \code{note}.
#' @export
reversible_jump_mcmc <- function(models, moves, init_model, init_theta = numeric(0),
                                 n_iter = 10000, burn_in = 0, thin = 1, seed = 0,
                                 within = NULL, within_scale = 0.5,
                                 within_weight = 1.0, move_weight = 1.0,
                                 jacobian = "analytic", keep_chain = TRUE) {
  if (!(jacobian %in% .JACOBIAN_ROUTES))
    stop(sprintf("bayrjmcmc: jacobian must be one of %s", paste(.JACOBIAN_ROUTES, collapse = ", ")))
  by_pair <- check_dimension_matching(models, moves)
  if (is.null(models[[init_model]]))
    stop(sprintf("bayrjmcmc: init_model %r is not a model", init_model))
  n_iter <- as.integer(n_iter)
  burn_in <- as.integer(burn_in)
  thin <- as.integer(thin)
  if (n_iter < 1)
    stop("bayrjmcmc: n_iter must be at least 1")
  if (burn_in < 0 || burn_in >= n_iter)
    stop("bayrjmcmc: burn_in must be in [0, n_iter)")
  if (thin < 1)
    stop("bayrjmcmc: thin must be at least 1")

  theta <- as.numeric(init_theta)
  if (length(theta) != as.integer(models[[init_model]]$dim))
    stop(sprintf("bayrjmcmc: init_theta has %d values but model %r has dim %d",
                 length(theta), init_model, as.integer(models[[init_model]]$dim)))

  avail <- list()
  for (name in names(models)) {
    opts <- list(list(kind = "within", mv = NULL, w = as.numeric(within_weight)))
    for (mv in moves)
      if (mv$frm == name)
        opts[[length(opts) + 1L]] <- list(kind = "jump", mv = mv,
                                          w = as.numeric(mv[["weight"]] %||% move_weight))
    tot <- sum(sapply(opts, function(o) o$w))
    if (tot <= 0.0)
      stop(sprintf("bayrjmcmc: model %r has no move with positive weight", name))
    avail[[name]] <- list(opts = opts, tot = tot)
  }

  uni <- .unif_stream(seed)
  cur <- init_model
  logp <- as.numeric(models[[cur]]$logpost(theta))
  visits <- new.env(hash = TRUE, parent = emptyenv())
  for (nm in names(models)) visits[[nm]] <- 0L
  tried <- new.env(hash = TRUE, parent = emptyenv())
  accepted <- new.env(hash = TRUE, parent = emptyenv())
  chain <- list()

  for (it in seq_len(n_iter)) {
    op <- avail[[cur]]
    opts <- op$opts; tot <- op$tot
    pick <- uni() * tot
    acc <- 0.0
    choice <- opts[[length(opts)]]
    for (k in seq_along(opts)) {
      acc <- acc + opts[[k]]$w
      if (pick < acc) { choice <- opts[[k]]; break }
    }
    kind <- choice$kind; mv <- choice$mv; w <- choice$w

    if (kind == "within") {
      label <- sprintf("within:%s", cur)
      tried[[label]] <- (tried[[label]] %||% 0L) + 1L
      if (length(theta) > 0L) {
        if (!is.null(within) && !is.null(within[[cur]])) {
          tmp <- within[[cur]](theta, uni)
          prop <- as.numeric(tmp[[1]]); log_ratio <- as.numeric(tmp[[2]])
        } else {
          prop <- .rw_within(theta, uni, within_scale)
          log_ratio <- 0.0
        }
        lp_new <- as.numeric(models[[cur]]$logpost(prop))
        if (log(uni()) < (lp_new - logp) + log_ratio) {
          theta <- prop; logp <- lp_new
          accepted[[label]] <- (accepted[[label]] %||% 0L) + 1L
        }
      }
    } else {
      label <- mv[["name"]] %||% sprintf("%s->%s", mv$frm, mv$to)
      tried[[label]] <- (tried[[label]] %||% 0L) + 1L
      u <- as.numeric(mv$propose(theta, uni))
      if (length(u) != as.integer(mv$n_u))
        stop(sprintf("bayrjmcmc: move %s proposed %d values of u but declares n_u = %d",
                     label, length(u), as.integer(mv$n_u)))
      tmp <- mv$transform(theta, u)
      theta2 <- as.numeric(tmp[[1]]); u2 <- as.numeric(tmp[[2]])
      dim2 <- as.integer(models[[mv$to]]$dim)
      if (length(theta2) != dim2 || length(u2) != as.integer(mv$n_u_rev))
        stop(sprintf("bayrjmcmc: move %s produced theta of length %d and u2 of length %d; the model has dim %d and the move declares n_u_rev = %d",
                     label, length(theta2), length(u2), dim2, as.integer(mv$n_u_rev)))

      if (jacobian == "numeric" || is.null(mv[["logjac"]])) {
        if (jacobian == "analytic" && is.null(mv[["logjac"]])) {
          logjac <- 0.0
        } else {
          n_from <- length(theta)
          tf <- mv$transform
          flat <- function(z) {
            a <- tf(z[seq_len(n_from)], z[(n_from + 1L):length(z)])
            c(as.numeric(a[[1]]), as.numeric(a[[2]]))
          }
          logjac <- numeric_log_jacobian(flat, c(theta, u))
        }
      } else {
        logjac <- as.numeric(mv[["logjac"]](theta, u, theta2, u2))
      }

      rev_key <- paste(mv$to, mv$frm, sep = "|")
      rev <- by_pair[[rev_key]]
      op2 <- avail[[mv$to]]
      tot2 <- op2$tot
      w_rev <- as.numeric(rev[["weight"]] %||% move_weight)
      log_j_from <- log(w) - log(tot)
      log_j_to <- log(w_rev) - log(tot2)

      logq_u <- if (!is.null(mv[["logq"]])) as.numeric(mv[["logq"]](theta, u)) else 0.0
      logq_rev <- if (!is.null(mv[["logq_rev"]])) as.numeric(mv[["logq_rev"]](theta2, u2)) else 0.0

      lp_new <- as.numeric(models[[mv$to]]$logpost(theta2))
      log_alpha <- rj_log_acceptance(logp, lp_new, log_j_from, log_j_to,
                                     logq_u, logq_rev, logjac)
      if (log(uni()) < log_alpha) {
        cur <- mv$to; theta <- theta2; logp <- lp_new
        accepted[[label]] <- (accepted[[label]] %||% 0L) + 1L
      }
    }

    if (it > burn_in) {
      visits[[cur]] <- visits[[cur]] + 1L
      if (keep_chain && ((it - burn_in - 1L) %% thin) == 0L)
        chain[[length(chain) + 1L]] <- list(model = cur, theta = list(theta))
    }
  }

  kept <- sum(unlist(as.list(visits)))
  freq <- list()
  for (nm in names(models)) freq[[nm]] <- visits[[nm]] / kept
  rates <- list()
  tnames <- ls(tried)
  for (k in tnames) {
    tv <- tried[[k]]
    if (tv > 0) rates[[k]] <- (accepted[[k]] %||% 0L) / tv
  }
  list(
    model_freq = freq,
    visits = as.list(visits),
    n_kept = kept,
    accept = rates,
    tried = as.list(tried),
    chain = chain,
    jacobian = jacobian,
    method = "reversible-jump MCMC, Green (1995) eq. 7; hybrid sampler with detailed balance within each move type",
    note = "dimension matching n1 + m1 == n2 + m2 is enforced for every move; j(x) is computed from the move weights available in each model, in both directions, because eq. 7 needs it at x' as well as x"
  )
}

#' step_function_loglik
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param y See Usage.
#' @param s Coerced to numeric by the body, with \code{as.numeric}.
#' @param h A vector; its length is taken and its elements indexed.
#' @param L Coerced to numeric by the body, with \code{as.numeric}.
#' @return The value of \code{out}, as built in the body.
#' @export
step_function_loglik <- function(y, s, h, L) {
  edges <- c(0.0, as.numeric(s), as.numeric(L))
  if (length(h) != length(edges) - 1L)
    stop(sprintf("bayrjmcmc: %d heights for %d intervals",
                 length(h), length(edges) - 1L))
  counts <- rep(0L, length(h))
  for (v in y) {
    v <- as.numeric(v)
    if (v < 0.0 || v > as.numeric(L))
      stop(sprintf("bayrjmcmc: point %g lies outside [0, %g]", v, as.numeric(L)))
    j <- 1L
    while (j + 1L < length(edges) - 1L && v >= edges[j + 1L]) j <- j + 1L
    counts[j] <- counts[j] + 1L
  }
  out <- 0.0
  for (j in seq_along(h)) {
    hj <- as.numeric(h[j])
    if (hj <= 0.0) return(-Inf)
    if (counts[j] > 0L) out <- out + counts[j] * log(hj)
    out <- out - hj * (edges[j + 1L] - edges[j])
  }
  out
}

#' changepoint_move_probabilities
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param lam Numeric; combined arithmetically in the body.
#' @param k_max A count; the body uses it as \code{seq_len(...)}.
#' @param cap Numeric; combined arithmetically in the body. Defaults to \code{0.9}.
#' @return A list with \code{eta}, \code{pi}, \code{b}, \code{d}, \code{c}.
#' @export
changepoint_move_probabilities <- function(lam, k_max, cap = 0.9) {
  lam <- as.numeric(lam)
  k_max <- as.integer(k_max)
  if (lam <= 0.0) stop("bayrjmcmc: lam must be positive")
  if (k_max < 1) stop("bayrjmcmc: k_max must be at least 1")
  raw_b <- sapply(seq_len(k_max + 1L) - 1L, function(k) min(1.0, lam / (k + 1.0)))
  raw_d <- c(0.0, sapply(seq_len(k_max), function(k) min(1.0, k / lam)))
  raw_b[k_max + 1L] <- 0.0
  worst <- max(raw_b + raw_d)
  cfac <- if (worst > 0.0) cap / worst else cap
  b <- cfac * raw_b
  d <- cfac * raw_d
  eta <- numeric(k_max + 1L); pi_ <- numeric(k_max + 1L)
  for (k in seq_len(k_max + 1L) - 1L) {
    rest <- 1.0 - b[k + 1L] - d[k + 1L]
    if (k == 0L) { eta[k + 1L] <- rest; pi_[k + 1L] <- 0.0 }
    else { eta[k + 1L] <- 0.5 * rest; pi_[k + 1L] <- 0.5 * rest }
  }
  list(eta = eta, pi = pi_, b = b, d = d, c = cfac)
}

#' birth_split_heights
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param h_j Coerced to numeric by the body, with \code{as.numeric}.
#' @param u Coerced to numeric by the body, with \code{as.numeric}.
#' @param s_left Coerced to numeric by the body, with \code{as.numeric}.
#' @param s_star Coerced to numeric by the body, with \code{as.numeric}.
#' @param s_right Coerced to numeric by the body, with \code{as.numeric}.
#' @return A vector, from \code{c}.
#' @export
birth_split_heights <- function(h_j, u, s_left, s_star, s_right) {
  span <- as.numeric(s_right) - as.numeric(s_left)
  if (span <= 0.0)
    stop("bayrjmcmc: empty interval in a birth move")
  w1 <- (as.numeric(s_star) - as.numeric(s_left)) / span
  r <- as.numeric(u) / (1.0 - as.numeric(u))
  lr <- log(r)
  hj <- as.numeric(h_j) * exp(-(1.0 - w1) * lr)
  hj1 <- as.numeric(h_j) * exp(w1 * lr)
  c(hj, hj1)
}

#' birth_log_jacobian
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param h_j Coerced to numeric by the body, with \code{as.numeric}.
#' @param h_new_left Coerced to numeric by the body, with \code{as.numeric}.
#' @param h_new_right Coerced to numeric by the body, with \code{as.numeric}.
#' @return A numeric value.
#' @export
birth_log_jacobian <- function(h_j, h_new_left, h_new_right) {
  2.0 * log(as.numeric(h_new_left) + as.numeric(h_new_right)) - log(as.numeric(h_j))
}

#' .merge_height
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param s_left Coerced to numeric by the body, with \code{as.numeric}.
#' @param s_mid Coerced to numeric by the body, with \code{as.numeric}.
#' @param s_right Coerced to numeric by the body, with \code{as.numeric}.
#' @param h_left Numeric; passed to \code{log}.
#' @param h_right Numeric; passed to \code{log}.
#' @return A numeric value.
#' @export
.merge_height <- function(s_left, s_mid, s_right, h_left, h_right) {
  span <- as.numeric(s_right) - as.numeric(s_left)
  exp(((as.numeric(s_mid) - as.numeric(s_left)) * log(h_left)
       + (as.numeric(s_right) - as.numeric(s_mid)) * log(h_right)) / span)
}

#' .log_k_prior_ratio
#'
#' A step of the bayrjmcmc_native implementation. Called by \code{changepoint_rjmcmc}.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param lam Numeric; passed to \code{log}.
#' @param k Numeric; combined arithmetically in the body.
#' @return A numeric value.
#' @export
.log_k_prior_ratio <- function(lam, k) {
  log(lam) - log(k + 1.0)
}

#' changepoint_rjmcmc
#'
#' A step of the bayrjmcmc_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param y Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{numeric(0)}.
#' @param L Numeric; passed to \code{log}. Defaults to \code{1}.
#' @param n_iter A count; the body uses it as \code{seq_len(...)}. Defaults to \code{40000}.
#' @param burn_in Numeric; combined arithmetically in the body. Defaults to \code{4000}.
#' @param lam Passed to \code{.log_k_prior_ratio}. Defaults to \code{3}.
#' @param k_max Numeric; combined arithmetically in the body. Defaults to \code{30}.
#' @param alpha Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param beta Numeric; passed to \code{log}. Defaults to \code{200}.
#' @param seed Passed to \code{.unif_stream}. Defaults to \code{0}.
#' @param cap Passed to \code{changepoint_move_probabilities}. Defaults to \code{0.9}.
#' @param use_likelihood A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param k_init A count; the body uses it as \code{seq_len(...)}. Defaults to \code{0}.
#' @param thin Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param keep_chain A flag; the body branches on it. Defaults to \code{FALSE}.
#' @return A list with \code{k_posterior}, \code{k_counts}, \code{k_mean}, \code{s}, \code{h}, \code{accept}, \code{tried}, \code{c}, \code{b}, \code{d}, \code{eta}, \code{pi}, \code{mean_s1_given_k1}, \code{var_s1_given_k1}, \code{mean_height}, \code{chain}, \code{n_kept}, \code{use_likelihood}, \code{method}, \code{note}.
#' @export
changepoint_rjmcmc <- function(y = numeric(0), L = 1.0, n_iter = 40000,
                               burn_in = 4000, lam = 3.0, k_max = 30,
                               alpha = 1.0, beta = 200.0, seed = 0, cap = 0.9,
                               use_likelihood = TRUE, k_init = 0, thin = 1,
                               keep_chain = FALSE) {
  L <- as.numeric(L)
  if (L <= 0.0) stop("bayrjmcmc: L must be positive")
  alpha <- as.numeric(alpha); beta <- as.numeric(beta)
  if (alpha <= 0.0 || beta <= 0.0)
    stop("bayrjmcmc: alpha and beta must be positive")
  n_iter <- as.integer(n_iter); burn_in <- as.integer(burn_in)
  if (n_iter < 1) stop("bayrjmcmc: n_iter must be at least 1")
  if (burn_in < 0 || burn_in >= n_iter)
    stop("bayrjmcmc: burn_in must be in [0, n_iter)")
  thin <- as.integer(thin)
  if (thin < 1) stop("bayrjmcmc: thin must be at least 1")
  y <- as.numeric(y)
  for (v in y)
    if (v < 0.0 || v > L)
      stop(sprintf("bayrjmcmc: point %g lies outside [0, %g]", v, L))
  mp <- changepoint_move_probabilities(lam, k_max, cap = cap)
  eta <- mp$eta; pi_ <- mp$pi; b <- mp$b; d <- mp$d; cfac <- mp$c
  k_init <- as.integer(k_init)
  if (k_init < 0 || k_init > k_max)
    stop("bayrjmcmc: k_init must be in [0, k_max]")

  uni <- .unif_stream(seed)
  s <- sapply(seq_len(k_init), function(i) L * (i) / (k_init + 1.0))
  h <- rep(alpha / beta, k_init + 1L)

  loglik_fn <- function(s_, h_) {
    if (use_likelihood) step_function_loglik(y, s_, h_, L) else 0.0
  }
  cur_ll <- loglik_fn(s, h)
  counts <- rep(0L, k_max + 1L)
  tried <- new.env(hash = TRUE, parent = emptyenv())
  for (k in c("height", "position", "birth", "death")) tried[[k]] <- 0L
  acc <- new.env(hash = TRUE, parent = emptyenv())
  for (k in c("height", "position", "birth", "death")) acc[[k]] <- 0L
  chain <- list()
  s1_sum <- 0.0; s1_sq <- 0.0; s1_n <- 0L
  h_sum <- 0.0; h_n <- 0L

  for (it in seq_len(n_iter)) {
    k <- length(s)
    pick <- uni()
    edges <- c(0.0, s, L)

    if (pick < b[k + 1L]) {
      tried[["birth"]] <- tried[["birth"]] + 1L
      s_star <- L * uni()
      j <- 1L
      while (j + 1L < length(edges) - 1L && s_star >= edges[j + 1L]) j <- j + 1L
      u <- uni()
      h_split <- birth_split_heights(h[j], u, edges[j], s_star, edges[j + 1L])
      hl <- h_split[1]; hr <- h_split[2]
      s_new <- c(s[seq_len(j - 1L)], s_star, s[seq.int(j, length(s))])
      h_new <- c(h[seq_len(j - 1L)], hl, hr, h[seq.int(j + 1L, length(h))])
      new_ll <- loglik_fn(s_new, h_new)

      log_prior <- (
        .log_k_prior_ratio(lam, k)
        + log(2.0 * (k + 1.0) * (2.0 * k + 3.0)) - 2.0 * log(L)
        + log((s_star - edges[j]) * (edges[j + 1L] - s_star)
              / (edges[j + 1L] - edges[j]))
        + alpha * log(beta) - lgamma(alpha)
        + (alpha - 1.0) * log(hl * hr / h[j])
        - beta * (hl + hr - h[j]))
      log_prop <- log(d[k + 2L]) + log(L) - log(b[k + 1L]) - log(k + 1.0)
      log_jac <- birth_log_jacobian(h[j], hl, hr)
      log_alpha <- (new_ll - cur_ll) + log_prior + log_prop + log_jac
      if (k + 1L <= k_max && log(uni()) < log_alpha) {
        s <- s_new; h <- h_new; cur_ll <- new_ll
        acc[["birth"]] <- acc[["birth"]] + 1L
      }
    } else if (pick < b[k + 1L] + d[k + 1L]) {
      tried[["death"]] <- tried[["death"]] + 1L
      i <- as.integer(uni() * k)
      if (i >= k) i <- k - 1L
      s_new <- c(s[seq_len(i - 1L)], s[seq.int(i + 1L, length(s))])
      h_merged <- .merge_height(edges[i + 1L], edges[i + 2L], edges[i + 3L],
                                h[i + 1L], h[i + 2L])
      h_new <- c(h[seq_len(i - 1L)], h_merged, h[seq.int(i + 2L, length(h))])
      new_ll <- loglik_fn(s_new, h_new)
      kk <- k - 1L
      s_star <- edges[i + 2L]
      left <- edges[i + 1L]
      right <- edges[i + 3L]
      log_prior <- (
        .log_k_prior_ratio(lam, kk)
        + log(2.0 * (kk + 1.0) * (2.0 * kk + 3.0)) - 2.0 * log(L)
        + log((s_star - left) * (right - s_star) / (right - left))
        + alpha * log(beta) - lgamma(alpha)
        + (alpha - 1.0) * log(h[i + 1L] * h[i + 2L] / h_merged)
        - beta * (h[i + 1L] + h[i + 2L] - h_merged))
      log_prop <- log(d[kk + 2L]) + log(L) - log(b[kk + 1L]) - log(kk + 1.0)
      log_jac <- birth_log_jacobian(h_merged, h[i + 1L], h[i + 2L])
      log_alpha <- -((cur_ll - new_ll) + log_prior + log_prop + log_jac)
      if (log(uni()) < log_alpha) {
        s <- s_new; h <- h_new; cur_ll <- new_ll
        acc[["death"]] <- acc[["death"]] + 1L
      }
    } else if (pick < b[k + 1L] + d[k + 1L] + eta[k + 1L]) {
      tried[["height"]] <- tried[["height"]] + 1L
      j <- as.integer(uni() * (k + 1L))
      if (j > k) j <- k
      hj <- h[j + 1L] * exp(uni() - 0.5)
      h_new <- h
      h_new[j + 1L] <- hj
      new_ll <- loglik_fn(s, h_new)
      log_alpha <- ((new_ll - cur_ll) + alpha * log(hj / h[j + 1L]) - beta * (hj - h[j + 1L]))
      if (log(uni()) < log_alpha) {
        h <- h_new; cur_ll <- new_ll
        acc[["height"]] <- acc[["height"]] + 1L
      }
    } else if (k >= 1L) {
      tried[["position"]] <- tried[["position"]] + 1L
      j <- as.integer(uni() * k)
      if (j >= k) j <- k - 1L
      lo <- edges[j + 1L]
      hi <- edges[j + 3L]
      s_star <- lo + (hi - lo) * uni()
      s_new <- s
      s_new[j + 1L] <- s_star
      new_ll <- loglik_fn(s_new, h)
      log_alpha <- (new_ll - cur_ll
                    + log((hi - s_star) * (s_star - lo)
                          / ((hi - s[j + 1L]) * (s[j + 1L] - lo))))
      if (log(uni()) < log_alpha) {
        s <- s_new; cur_ll <- new_ll
        acc[["position"]] <- acc[["position"]] + 1L
      }
    }

    if (it > burn_in) {
      counts[length(s) + 1L] <- counts[length(s) + 1L] + 1L
      if (length(s) == 1L) {
        s1_sum <- s1_sum + s[1L]
        s1_sq <- s1_sq + s[1L] * s[1L]
        s1_n <- s1_n + 1L
      }
      for (v in h) { h_sum <- h_sum + v; h_n <- h_n + 1L }
      if (keep_chain && ((it - burn_in - 1L) %% thin) == 0L)
        chain[[length(chain) + 1L]] <- list(s = list(s), h = list(h))
    }
  }

  kept <- sum(counts)
  k_post <- counts / kept
  rates <- list()
  for (key in c("height", "position", "birth", "death")) {
    tv <- tried[[key]]
    if (tv > 0L) rates[[key]] <- acc[[key]] / tv
  }
  list(
    k_posterior = k_post,
    k_counts = counts,
    k_mean = sum((seq_along(k_post) - 1L) * k_post),
    s = list(s),
    h = list(h),
    accept = rates,
    tried = as.list(tried),
    c = cfac,
    b = b,
    d = d,
    eta = eta,
    pi = pi_,
    mean_s1_given_k1 = if (s1_n > 0L) s1_sum / s1_n else NaN,
    var_s1_given_k1 = if (s1_n > 1L) (s1_sq / s1_n - (s1_sum / s1_n)^2) else NaN,
    mean_height = if (h_n > 0L) h_sum / h_n else NaN,
    chain = chain,
    n_kept = kept,
    use_likelihood = as.logical(use_likelihood),
    method = "Green (1995) §4 change-point sampler: height, position, birth and death moves on a step-function rate, with the §4-3 acceptance ratios",
    note = "with use_likelihood=False the target is the prior, so k must come back Poisson(lam) truncated at k_max and the heights Gamma(alpha, beta)"
  )
}

bayrjmcmc <- reversible_jump_mcmc

#' morie_bayrjmcmc
#'
#' A step of the bayrjmcmc_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' the source it follows.
#'
#' @param ... Passed through.
#' @return The value of \code{reversible_jump_mcmc}.
#' @export
morie_bayrjmcmc <- function(...) reversible_jump_mcmc(...)
