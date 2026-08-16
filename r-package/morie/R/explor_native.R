# morie.fn -- function file (rootcoder007/morie)
# Intrinsic Curiosity Module: curiosity as forward-model error in a
# learned, action-relevant feature space.
#
# References
# Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017)
# "Curiosity-driven Exploration by Self-supervised Prediction", ICML,
# arXiv:1705.05363. Eqs. 2-7.

.explor_EPS <- 1e-300

#' .mat
#'
#' A step of the explor_native implementation. Called by \code{explor}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A matrix; passed to \code{nrow}.
#' @param name See Usage.
#' @return The value of \code{X}, as built in the body.
#' @export
.mat <- function(x, name) {
  if (is.data.frame(x)) x <- as.matrix(x)
  x <- as.matrix(x)
  if (nrow(x) == 0L || ncol(x) == 0L) stop(sprintf("explor: %s must be non-empty", name))
  X <- matrix(as.numeric(x), nrow = nrow(x))
  X
}

#' .matvec
#'
#' A step of the explor_native implementation. Called by \code{explor}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param W A matrix; indexed by row and column.
#' @param x A vector; its length is taken and its elements indexed.
#' @return The value of \code{out}, as built in the body.
#' @export
.matvec <- function(W, x) {
  n_out <- ncol(W)
  out <- rep(0, n_out)
  for (j in seq_along(x)) {
    xj <- x[j]
    if (xj == 0) next
    row <- W[j, ]
    for (o in seq_len(n_out)) out[o] <- out[o] + row[o] * xj
  }
  out
}

#' .explor_softmax
#'
#' A step of the explor_native implementation. Called by \code{explor}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param z Numeric; passed to \code{max}.
#' @return A numeric value.
#' @export
.explor_softmax <- function(z) {
  m <- max(z)
  e <- exp(z - m)
  s <- sum(e)
  e / s
}

#' explor
#'
#' A step of the explor_native implementation. Called by \code{morie_explor}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param states Passed to \code{.mat}.
#' @param actions Passed to \code{.mat}.
#' @param next_states Passed to \code{.mat}.
#' @param n_actions Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.
#' @param n_features Coerced to integer by the body, with \code{as.integer}. Defaults to \code{8L}.
#' @param eta Numeric; combined arithmetically in the body. Defaults to \code{1}.
#' @param beta Numeric; combined arithmetically in the body. Defaults to \code{0.2}.
#' @param lr Numeric; combined arithmetically in the body. Defaults to \code{0.05}.
#' @param epochs Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1L}.
#' @param features One of \code{"identity"}, \code{"inverse"}. Defaults to \code{"inverse"}.
#' @param discrete A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param seed Passed to \code{.ghc_rng}. Defaults to \code{0L}.
#' @return The value of \code{payload}, as built in the body.
#' @export
explor <- function(states, actions, next_states, n_actions = NULL,
                   n_features = 8L, eta = 1.0, beta = 0.2, lr = 0.05,
                   epochs = 1L, features = "inverse", discrete = TRUE,
                   seed = 0L) {
  if (!(features %in% c("inverse", "identity"))) {
    stop(sprintf("explor: features must be one of %s, got %s",
                 paste(sQuote(c("inverse", "identity")), collapse = ", "),
                 sQuote(features)))
  }
  eta <- as.numeric(eta)
  if (!(eta > 0)) stop("explor: eta must be > 0")
  beta <- as.numeric(beta)
  if (beta < 0 || beta > 1) stop("explor: beta must lie in [0, 1]")

  S <- .mat(states, "states")
  S1 <- .mat(next_states, "next_states")
  if (nrow(S) != nrow(S1)) {
    stop("explor: states and next_states must have the same length")
  }
  if (ncol(S) != ncol(S1)) {
    stop("explor: states and next_states must have the same width")
  }
  T <- nrow(S)
  d <- ncol(S)

  if (isTRUE(discrete)) {
    Avec <- as.integer(as.vector(actions))
    if (length(Avec) != T) {
      stop(sprintf("explor: got %d actions for %d transitions",
                   length(Avec), T))
    }
    nA <- if (is.null(n_actions)) max(Avec) + 1L else as.integer(n_actions)
    if (nA < 2L) stop("explor: need at least 2 discrete actions")
    if (min(Avec) < 0L || max(Avec) >= nA) {
      stop("explor: action index out of range")
    }
    a_dim <- nA
  } else {
    Ac <- .mat(actions, "actions")
    if (nrow(Ac) != T) {
      stop(sprintf("explor: got %d actions for %d transitions",
                   nrow(Ac), T))
    }
    a_dim <- ncol(Ac)
  }

  rng <- .ghc_rng(seed)
  if (features == "identity") {
    k <- d
    Wphi <- NULL
  } else {
    k <- as.integer(n_features)
    if (k < 1L) stop("explor: n_features must be >= 1")
    s <- 0.1 / sqrt(d)
    raw <- .ghc_unif(rng, d * k)
    Wphi <- matrix(raw * 2 * s - s, nrow = d, ncol = k)
  }

  phi <- function(x) {
    if (is.null(Wphi)) return(as.numeric(x))
    v <- .matvec(Wphi, as.numeric(x))
    tanh(v)
  }

  Winv <- matrix(0, nrow = 2L * k, ncol = a_dim)
  Wfwd <- matrix(0, nrow = k + a_dim, ncol = k)

  curve <- numeric(0)
  for (ep in seq_len(max(1L, as.integer(epochs)))) {
    rewards <- numeric(T)
    lf_tot <- 0
    li_tot <- 0
    n_correct <- 0L
    for (t in seq_len(T)) {
      p <- phi(S[t, ])
      p1 <- phi(S1[t, ])
      if (isTRUE(discrete)) {
        avec <- rep(0, a_dim)
        avec[Avec[t] + 1L] <- 1
      } else {
        avec <- Ac[t, ]
      }

      inp_i <- c(p, p1)
      zi <- .matvec(Winv, inp_i)
      if (isTRUE(discrete)) {
        pr <- .explor_softmax(zi)
        li <- -log(max(pr[Avec[t] + 1L], .explor_EPS))
        gi <- pr - avec
        if (which.max(pr) - 1L == Avec[t]) n_correct <- n_correct + 1L
      } else {
        li <- 0.5 * sum((zi - avec) ^ 2)
        gi <- zi - avec
      }
      li_tot <- li_tot + li

      inp_f <- c(p, avec)
      ph <- .matvec(Wfwd, inp_f)
      ef <- ph - p1
      lf <- 0.5 * sum(ef * ef)
      lf_tot <- lf_tot + lf
      rewards[t] <- eta * lf

      if (!is.null(Wphi)) {
        dphi <- rep(0, 2L * k)
        for (j in seq_len(2L * k)) {
          acc <- 0
          row <- Winv[j, ]
          for (o in seq_len(a_dim)) acc <- acc + row[o] * gi[o]
          dphi[j] <- acc
        }
        for (half in c(0L, 1L)) {
          xin <- if (half == 0L) S[t, ] else S1[t, ]
          ph_ <- if (half == 0L) p else p1
          for (j in seq_len(k)) {
            g <- dphi[half * k + j] * (1 - ph_[j] * ph_[j])
            if (g == 0) next
            step <- lr * (1 - beta) * g
            for (dd in seq_len(d)) {
              if (xin[dd] != 0) Wphi[dd, j] <- Wphi[dd, j] - step * xin[dd]
            }
          }
        }
      }
      for (j in seq_len(2L * k)) {
        xj <- inp_i[j]
        if (xj == 0) next
        for (o in seq_len(a_dim)) {
          Winv[j, o] <- Winv[j, o] - lr * (1 - beta) * gi[o] * xj
        }
      }
      for (j in seq_len(k + a_dim)) {
        xj <- inp_f[j]
        if (xj == 0) next
        for (o in seq_len(k)) {
          Wfwd[j, o] <- Wfwd[j, o] - lr * beta * ef[o] * xj
        }
      }
    }
    curve <- c(curve, ((1 - beta) * li_tot + beta * lf_tot) / T)
  }

  n <- length(rewards)
  tenth <- max(1L, n %/% 10L)
  payload <- list(
    estimate = rewards,
    intrinsic_reward = rewards,
    forward_loss = as.numeric(lf_tot / T),
    inverse_loss = as.numeric(li_tot / T),
    objective = as.numeric(curve[length(curve)]),
    loss_curve = curve,
    phi = t(apply(S, 1, phi)),
    phi_next = t(apply(S1, 1, phi)),
    mean_first = as.numeric(sum(rewards[seq_len(tenth)]) / tenth),
    mean_last = as.numeric(sum(rewards[(n - tenth + 1L):n]) / tenth),
    eta = eta, beta = beta, n = n, features = features,
    method = "ICM (Pathak et al. 2017, eqs. 2-7)"
  )
  if (isTRUE(discrete)) {
    payload$inverse_accuracy <- as.numeric(n_correct) / T
  }
  payload
}

#' morie_explor
#'
#' A step of the explor_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param states See Usage.
#' @param actions See Usage.
#' @param next_states See Usage.
#' @param n_actions Defaults to \code{NULL}.
#' @param n_features Defaults to \code{8L}.
#' @param eta Defaults to \code{1}.
#' @param beta Defaults to \code{0.2}.
#' @param lr Defaults to \code{0.05}.
#' @param epochs Defaults to \code{1L}.
#' @param features Defaults to \code{"inverse"}.
#' @param discrete Defaults to \code{TRUE}.
#' @param seed Defaults to \code{0L}.
#' @return The value of \code{explor}.
#' @export
morie_explor <- function(states, actions, next_states, n_actions = NULL,
                         n_features = 8L, eta = 1.0, beta = 0.2,
                         lr = 0.05, epochs = 1L, features = "inverse",
                         discrete = TRUE, seed = 0L) {
  explor(states = states, actions = actions, next_states = next_states,
         n_actions = n_actions, n_features = n_features, eta = eta,
         beta = beta, lr = lr, epochs = epochs, features = features,
         discrete = discrete, seed = seed)
}
