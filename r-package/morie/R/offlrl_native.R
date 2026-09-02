# Sources: Kumar, A., Zhou, A., Tucker, G. & Levine, S. (2020)
# "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS,
# arXiv:2006.04779 (eqs. 1-4: the CQL penalty with the asymmetry
# between pushing Q DOWN under mu and UP under pi_beta, the three
# variants H/rho/mu, and Theorems 3.1-3.2 distinguishing the pointwise
# and the weaker expected-value lower bound).
#
# Native implementation mirroring Python morie.fn.offlrl exactly: the
# same dataset checks (transitions of length 4 or 5, with done
# defaults to FALSE), the same empirical behaviour policy read off the
# data, the same Bellman target under max or under pi, the same
# gradient split into a CQL term that pushes Q down under mu and up
# under pi_beta and a Bellman term on the data, the same final
# penalty and Bellman error, and the same exit condition on the
# largest Q update.

offlrl_variants <- c("H", "rho", "mu")
offlrl_backups <- c("max", "pi")

#' offlrl_logsumexp
#'
#' A step of the offlrl_native implementation. Called by \code{morie_offlrl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param v Numeric; passed to \code{max}.
#' @return A numeric value.
#' @export
offlrl_logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}

#' offlrl_softmax
#'
#' A step of the offlrl_native implementation. Called by \code{morie_offlrl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param v Numeric; passed to \code{max}.
#' @return A numeric value.
#' @export
offlrl_softmax <- function(v) {
  m <- max(v)
  e <- exp(v - m)
  e / sum(e)
}

#' offlrl_as_dist
#'
#' A step of the offlrl_native implementation. Called by \code{morie_offlrl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param d Optional; may be \code{NULL}. A vector; indexed elementwise.
#' @param S See Usage.
#' @param A See Usage.
#' @param name Passed to \code{stop}.
#' @return A list with \code{matrix}, \code{lookup}.
#' @export
offlrl_as_dist <- function(d, S, A, name) {
  if (is.null(d)) return(NULL)
  out <- list()
  for (s in S) {
    for (a in A) {
      if (is.function(d)) {
        out[[paste0(s, "|", a)]] <- as.numeric(d(s, a))
      } else {
        out[[paste0(s, "|", a)]] <- as.numeric(d[[paste0(s, "|", a)]])
      }
    }
  }
  for (s in S) {
    tot <- 0
    for (a in A) tot <- tot + out[[paste0(s, "|", a)]]
    if (abs(tot - 1) > 1e-6)
      stop("offlrl: ", name, "(.|", deparse(s), ") sums to ", tot,
           ", not 1")
  }
  list(matrix = out, lookup = function(s, a) {
    out[[paste0(s, "|", a)]]
  })
}

#' offlrl_lookup
#'
#' A step of the offlrl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param mat A vector; indexed elementwise.
#' @param s Passed to \code{paste0}.
#' @param a Passed to \code{paste0}.
#' @return The value of \code{[[}.
#' @export
offlrl_lookup <- function(mat, s, a) {
  mat[[paste0(s, "|", a)]]
}

#' offlrl_safe_max_key
#'
#' A step of the offlrl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param qmap A vector; indexed elementwise.
#' @param s Passed to \code{paste0}.
#' @param A A vector; indexed elementwise.
#' @return The value of \code{best_a}, as built in the body.
#' @export
offlrl_safe_max_key <- function(qmap, s, A) {
  best_v <- -Inf
  best_a <- A[1]
  for (a in A) {
    v <- qmap[[paste0(s, "|", a)]]
    if (v > best_v) { best_v <- v
    best_a <- a }
  }
  best_a
}

#' morie_offlrl
#'
#' A step of the offlrl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param dataset The body requires: offlrl: dataset must be non-empty.
#' @param states Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.
#' @param actions Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.
#' @param alpha Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param gamma Numeric; combined arithmetically in the body. Defaults to \code{0.99}.
#' @param variant One of \code{"H"}, \code{"mu"}, \code{"rho"}. Defaults to \code{"H"}.
#' @param backup One of \code{"max"}, \code{"pi"}. Defaults to \code{"max"}.
#' @param policy The body requires: offlrl: variant='rho' needs policy(a|s) to play the role of pi^\{k-1\}.
#' @param mu The body requires: offlrl: variant must be one of c('H','rho','mu'), got '.
#' @param lr Numeric; combined arithmetically in the body. Defaults to \code{0.5}.
#' @param iters Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2000}.
#' @param tol Passed to \code{<}. Defaults to \code{1e-12}.
#' @return A list with \code{estimate}, \code{q}, \code{value}, \code{greedy}, \code{behavior}, \code{counts}, \code{penalty}, \code{bellman_error}, \code{objective}, \code{alpha}, \code{variant}, \code{backup}, \code{n_transitions}, \code{method}.
#' @export
morie_offlrl <- function(dataset, states = NULL, actions = NULL,
                         alpha = 1.0, gamma = 0.99, variant = "H",
                         backup = "max", policy = NULL, mu = NULL,
                         lr = 0.5, iters = 2000, tol = 1e-12) {
  variant <- as.character(variant)
  backup <- as.character(backup)
  if (!(variant %in% offlrl_variants))
    stop("offlrl: variant must be one of c('H','rho','mu'), got '",
         variant, "'")
  if (!(backup %in% offlrl_backups))
    stop("offlrl: backup must be 'max' or 'pi', got '", backup, "'")
  alpha <- as.numeric(alpha)
  if (alpha < 0) stop("offlrl: alpha must be >= 0, got ", alpha)

  D <- list()
  for (t in dataset) {
    if (length(t) == 4L) {
      s <- t[[1]]
      a <- t[[2]]
      r <- as.numeric(t[[3]])
      s1 <- t[[4]]
      done <- FALSE
    } else if (length(t) == 5L) {
      s <- t[[1]]
      a <- t[[2]]
      r <- as.numeric(t[[3]])
      s1 <- t[[4]]
      done <- as.logical(t[[5]])
    } else {
      stop("offlrl: each transition must be (s, a, r, s_next) or (s, a, r, s_next, done)")
    }
    D[[length(D) + 1L]] <- list(s = s, a = a, r = r, s1 = s1,
                                done = done)
  }
  if (length(D) == 0L) stop("offlrl: dataset must be non-empty")

  if (is.null(states)) {
    seen <- character(0)
    for (t in D) {
      seen <- c(seen, as.character(t$s), as.character(t$s1))
    }
    S <- sort(unique(seen))
  } else {
    S <- as.list(states)
  }
  if (is.null(actions)) {
    seen <- character(0)
    for (t in D) seen <- c(seen, as.character(t$a))
    A <- sort(unique(seen))
  } else {
    A <- as.list(actions)
  }
  if (length(S) == 0L || length(A) == 0L)
    stop("offlrl: states and actions must be non-empty")

  n_sa <- list()
  n_s <- list()
  for (t in D) {
    k <- paste0(t$s, "|", t$a)
    n_sa[[k]] <- if (is.null(n_sa[[k]])) 1L else n_sa[[k]] + 1L
    sk <- as.character(t$s)
    n_s[[sk]] <- if (is.null(n_s[[sk]])) 1L else n_s[[sk]] + 1L
  }
  behavior <- list()
  for (s in S) {
    sk <- as.character(s)
    if (is.null(n_s[[sk]])) next
    for (a in A) {
      k <- paste0(s, "|", a)
      behavior[[k]] <- (if (is.null(n_sa[[k]])) 0 else n_sa[[k]]) /
        as.numeric(n_s[[sk]])
    }
  }
  pol <- offlrl_as_dist(policy, S, A, "policy")
  muu <- offlrl_as_dist(mu, S, A, "mu")
  if (variant == "mu" && is.null(muu))
    stop("offlrl: variant='mu' needs mu(a|s)")
  if (backup == "pi" && is.null(pol))
    stop("offlrl: backup='pi' needs policy(a|s)")
  if (variant == "rho" && is.null(pol))
    stop("offlrl: variant='rho' needs policy(a|s) to play the role of pi^{k-1}")

  Q <- list()
  for (s in S) for (a in A) Q[[paste0(s, "|", a)]] <- 0
  data_states <- Filter(function(s) !is.null(n_s[[as.character(s)]]), S)
  N <- as.numeric(length(D))

  for (it in seq_len(as.integer(iters))) {
    target <- list()
    cnt <- list()
    for (t in D) {
      k <- paste0(t$s, "|", t$a)
      if (isTRUE(t$done)) {
        tt <- t$r
      } else if (backup == "max") {
        best <- -Inf
        for (b in A) {
          v <- Q[[paste0(t$s1, "|", b)]]
          if (v > best) best <- v
        }
        tt <- t$r + gamma * best
      } else {
        acc <- 0
        for (b in A) acc <- acc + pol$lookup(t$s1, b) *
          Q[[paste0(t$s1, "|", b)]]
        tt <- t$r + gamma * acc
      }
      target[[k]] <- (if (is.null(target[[k]])) 0 else target[[k]]) + tt
      cnt[[k]] <- (if (is.null(cnt[[k]])) 0L else cnt[[k]]) + 1L
    }
    for (k in names(target)) target[[k]] <- target[[k]] / cnt[[k]]

    grad <- list()
    for (s in S) for (a in A) grad[[paste0(s, "|", a)]] <- 0
    for (s in data_states) {
      w <- n_s[[as.character(s)]] / N
      qs <- vapply(A, function(a) Q[[paste0(s, "|", a)]], numeric(1))
      if (variant == "H") {
        push <- offlrl_softmax(qs)
      } else if (variant == "rho") {
        m <- max(qs)
        e <- vapply(A, function(a) pol$lookup(s, a) *
                      exp(Q[[paste0(s, "|", a)]] - m), numeric(1))
        z <- sum(e)
        push <- if (z > 0) e / z else rep(1 / length(A), length(A))
      } else {
        push <- vapply(A, function(a) muu$lookup(s, a), numeric(1))
      }
      for (i in seq_along(A)) {
        b <- A[[i]]
        k <- paste0(s, "|", b)
        grad[[k]] <- grad[[k]] + alpha * w *
          (push[i] - (if (is.null(behavior[[k]])) 0 else behavior[[k]]))
      }
    }
    for (k in names(target)) {
      grad[[k]] <- grad[[k]] + (cnt[[k]] / N) * (Q[[k]] - target[[k]])
    }

    delta <- 0
    for (k in names(Q)) {
      step <- lr * grad[[k]]
      Q[[k]] <- Q[[k]] - step
      if (abs(step) > delta) delta <- abs(step)
    }
    if (delta < tol) break
  }

  value <- list()
  greedy <- list()
  for (s in S) {
    best_v <- -Inf
    best_a <- NULL
    for (a in A) {
      v <- Q[[paste0(s, "|", a)]]
      if (v > best_v) { best_v <- v
      best_a <- a }
    }
    value[[as.character(s)]] <- best_v
    greedy[[as.character(s)]] <- best_a
  }

  pen <- 0
  for (s in data_states) {
    w <- n_s[[as.character(s)]] / N
    qs <- vapply(A, function(a) Q[[paste0(s, "|", a)]], numeric(1))
    if (variant == "H") {
      first <- offlrl_logsumexp(qs)
    } else if (variant == "rho") {
      m <- max(qs)
      acc <- 0
      for (i in seq_along(A)) {
        b <- A[[i]]
        acc <- acc + pol$lookup(s, b) * exp(Q[[paste0(s, "|", b)]] - m)
      }
      first <- m + log(acc)
    } else {
      first <- 0
      for (a in A) first <- first + muu$lookup(s, a) *
        Q[[paste0(s, "|", a)]]
    }
    second <- 0
    for (a in A) {
      k <- paste0(s, "|", a)
      second <- second + (if (is.null(behavior[[k]])) 0 else behavior[[k]]) *
        Q[[k]]
    }
    pen <- pen + w * (first - second)
  }

  berr <- 0
  for (t in D) {
    if (isTRUE(t$done)) {
      tt <- t$r
    } else if (backup == "max") {
      best <- -Inf
      for (b in A) {
        v <- Q[[paste0(t$s1, "|", b)]]
        if (v > best) best <- v
      }
      tt <- t$r + gamma * best
    } else {
      acc <- 0
      for (b in A) acc <- acc + pol$lookup(t$s1, b) *
        Q[[paste0(t$s1, "|", b)]]
      tt <- t$r + gamma * acc
    }
    berr <- berr + 0.5 * (Q[[paste0(t$s, "|", t$a)]] - tt) ^ 2
  }
  berr <- berr / N

  Qdict <- list()
  for (s in S) for (a in A) {
    k <- paste0(s, "|", a)
    Qdict[[paste0(as.character(s), "_", as.character(a))]] <- Q[[k]]
  }
  list(estimate = Qdict, q = Qdict, value = value, greedy = greedy,
       behavior = behavior, counts = n_sa,
       penalty = as.numeric(pen), bellman_error = as.numeric(berr),
       objective = as.numeric(alpha * pen + berr),
       alpha = alpha, variant = variant, backup = backup,
       n_transitions = length(D),
       method = paste0("CQL (Kumar et al. 2020, eq. ",
                       if (variant %in% c("H", "rho")) "4" else "2", ")"))
}

offlrl <- morie_offlrl
offline_rl_cql <- morie_offlrl
offlinerlcql <- morie_offlrl
conservative_q_learning <- morie_offlrl

#' offlrl_cheatsheet
#'
#' A step of the offlrl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
offlrl_cheatsheet <- function() {
  paste("offlrl: CQL (Kumar 2020). Fitted Q plus alpha*(push DOWN ",
        "E_mu[Q] - push UP E_pi_beta[Q]) so the Q-function LOWER ",
        "BOUNDS the truth and OOD actions stop being over-estimated. ",
        "variant='H' is eq. 4's logsumexp (rho=Unif); 'rho' uses ",
        "pi^{k-1}; 'mu' is eq. 2 directly. Thm 3.2 bounds the ",
        "EXPECTED value under pi, not pointwise. alpha=0 is plain ",
        "fitted Q.")
}
