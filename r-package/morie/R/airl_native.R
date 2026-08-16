# Adversarial Inverse Reinforcement Learning.
# Sources: Fu, J., Luo, K. & Levine, S. (2018) "Learning Robust Rewards
# with Adversarial Inverse Reinforcement Learning", ICLR 2018,
# arXiv:1710.11248 -- eq. 4 (discriminator with the g(s) + gamma h(s')
# - h(s) shaping, so Theorem C.1 can recover the reward rather than
# the advantage), Algorithm 1 (alternating fit D and use D to define
# the reward r = log D - log(1 - D)), and Theorem C.1 (g* = r + const
# and h* = V* + const under deterministic dynamics and a state-only
# ground truth reward).
# Maximum-entropy IRL: the soft value iteration we use to validate
# the recovery is the MaxEnt MDP of the same paper.
#
# Native implementation mirroring morie.fn.airl exactly: the same
# discriminator parameterisation (state-only by default, (s, a)
# otherwise), the same per-row logistic gradient compressed by row
# (identical transitions are tallied, not stored), the same line-6
# reward. No randomness is used here; the function is deterministic.

#' .airl_log
#'
#' A step of the airl_native implementation. Called by \code{airl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Numeric; passed to \code{max}.
#' @param floor Numeric; passed to \code{max}. Defaults to \code{1e-300}.
#' @return A numeric value.
#' @export
.airl_log <- function(x, floor = 1e-300) log(max(x, floor))

#' .airl_key
#'
#' A step of the airl_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param s A vector; its length is taken.
#' @return One of two values, depending on the branch taken.
#' @export
.airl_key <- function(s) {
  if (is.numeric(s) && length(s) == 1L) s
  else as.numeric(s)
}

#' .airl_prep
#'
#' A step of the airl_native implementation. Called by \code{airl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param S A vector; its length is taken.
#' @param A A vector; its length is taken.
#' @param S1 A vector; its length is taken.
#' @param LP A vector; its length is taken.
#' @param name See Usage.
#' @return A list with \code{S}, \code{A}, \code{S1}, \code{LP}.
#' @export
.airl_prep <- function(S, A, S1, LP, name) {
  if (length(S) == 0L || length(A) == 0L || length(S1) == 0L ||
        length(LP) == 0L)
    stop("airl: ", name, " states, actions, next states and log_policy ",
         "must be non-empty and the same length")
  if (!(length(S) == length(A) && length(A) == length(S1) &&
        length(S1) == length(LP)))
    stop("airl: ", name, " states, actions, next states and log_policy ",
         "must be non-empty and the same length")
  list(S = lapply(S, .airl_key),
       A = lapply(A, function(a) a),
       S1 = lapply(S1, .airl_key),
       LP = as.numeric(LP))
}

#' Adversarial Inverse Reinforcement Learning
#'
#' Fit the AIRL discriminator and return the recovered reward,
#' following Fu, Luo & Levine (2018) eq. 4 and Algorithm 1.
#' @param expert_states,expert_actions,expert_next Expert transitions.
#' @param expert_log_policy log pi(a|s) under the *current* policy.
#' @param policy_states,policy_actions,policy_next Policy transitions.
#' @param policy_log_policy log pi(a|s) for the policy transitions.
#' @param gamma Discount used in the shaping term.
#' @param state_only TRUE: parameterise g on s alone (Theorem C.1);
#'   FALSE: g(s, a).
#' @param lr,epochs,l2 Full-batch gradient ascent on the logistic
#'   log-likelihood with optional ridge penalty.
#' @return List with reward (line 6 of Algorithm 1), g, h, f_policy,
#'   f_expert, D_policy, D_expert, accuracy, log_likelihood, gamma,
#'   state_only, method.
#' @export
airl <- function(expert_states, expert_actions, expert_next,
                 expert_log_policy, policy_states, policy_actions,
                 policy_next, policy_log_policy, gamma = 0.99,
                 state_only = TRUE, lr = 0.1, epochs = 500L, l2 = 0) {
  Eprep <- .airl_prep(expert_states, expert_actions, expert_next,
                      expert_log_policy, "expert")
  Pprep <- .airl_prep(policy_states, policy_actions, policy_next,
                      policy_log_policy, "policy")
  nE <- length(Eprep$S); nP <- length(Pprep$S)
  Es <- lapply(seq_len(nE), function(k) list(S = Eprep$S[[k]],
                                             A = Eprep$A[[k]],
                                             S1 = Eprep$S1[[k]],
                                             LP = Eprep$LP[k]))
  Ps <- lapply(seq_len(nP), function(k) list(S = Pprep$S[[k]],
                                             A = Pprep$A[[k]],
                                             S1 = Pprep$S1[[k]],
                                             LP = Pprep$LP[k]))
  all_states <- unique(c(lapply(Es, function(t) t$S),
                         lapply(Es, function(t) t$S1),
                         lapply(Ps, function(t) t$S),
                         lapply(Ps, function(t) t$S1)))
  if (state_only) {
    gkeys <- all_states
  } else {
    gkeys <- unique(c(lapply(Es, function(t) list(t$S, t$A)),
                      lapply(Ps, function(t) list(t$S, t$A))))
  }
  gi <- new.env(hash = TRUE, parent = emptyenv())
  for (i in seq_along(gkeys)) assign(deparse(gkeys[[i]], control = "useSource"),
                                     i, envir = gi)
  hi <- new.env(hash = TRUE, parent = emptyenv())
  for (i in seq_along(all_states)) assign(deparse(all_states[[i]],
                                                  control = "useSource"),
                                          i, envir = hi)
  ng <- length(gkeys); nh <- length(all_states)
  g <- rep(0, ng); h <- rep(0, nh)
  gamma <- as.numeric(gamma); lr <- as.numeric(lr); l2 <- as.numeric(l2)

  gkey_of <- function(t) if (state_only) t$S else list(t$S, t$A)
  gi_idx <- function(k)
    get(deparse(k, control = "useSource"), envir = gi, inherits = FALSE)
  hi_idx <- function(k)
    get(deparse(k, control = "useSource"), envir = hi, inherits = FALSE)
  f_of <- function(t) g[gi_idx(gkey_of(t))] + gamma * h[hi_idx(t$S1)] -
    h[hi_idx(t$S)]
  d_of <- function(t) {
    z <- f_of(t) - t$LP
    if (z >= 0) 1 / (1 + exp(-z)) else { ez <- exp(z); ez / (1 + ez) }
  }

  # Compress to unique transitions: identical rows contribute identical
  # gradients, so the fit is unchanged and the cost stays bounded.
  Ec <- list(); Pc <- list()
  Ecount <- new.env(hash = TRUE, parent = emptyenv())
  for (t in Es) {
    key <- paste0(deparse(t$S, control = "useSource"), "|",
                  deparse(t$A, control = "useSource"), "|",
                  deparse(t$S1, control = "useSource"), "|", t$LP)
    assign(key, (get(key, envir = Ecount, inherits = FALSE) %||% 0) + 1,
           envir = Ecount)
  }
  Ekeys <- ls(Ecount)
  for (k in Ekeys) {
    parts <- strsplit(k, "|", fixed = TRUE)[[1]]
    Ec[[length(Ec) + 1L]] <- list(t = list(S = .airl_key_from_str(parts[1]),
                                           A = .airl_key_from_str(parts[2]),
                                           S1 = .airl_key_from_str(parts[3]),
                                           LP = as.numeric(parts[4])),
                                  w = get(k, envir = Ecount) / nE)
  }
  Pcount <- new.env(hash = TRUE, parent = emptyenv())
  for (t in Ps) {
    key <- paste0(deparse(t$S, control = "useSource"), "|",
                  deparse(t$A, control = "useSource"), "|",
                  deparse(t$S1, control = "useSource"), "|", t$LP)
    assign(key, (get(key, envir = Pcount, inherits = FALSE) %||% 0) + 1,
           envir = Pcount)
  }
  Pkeys <- ls(Pcount)
  for (k in Pkeys) {
    parts <- strsplit(k, "|", fixed = TRUE)[[1]]
    Pc[[length(Pc) + 1L]] <- list(t = list(S = .airl_key_from_str(parts[1]),
                                           A = .airl_key_from_str(parts[2]),
                                           S1 = .airl_key_from_str(parts[3]),
                                           LP = as.numeric(parts[4])),
                                  w = get(k, envir = Pcount) / nP)
  }

  for (ep in seq_len(max(1L, as.integer(epochs)))) {
    dg <- rep(0, ng); dh <- rep(0, nh)
    for (row in Ec) {
      t <- row$t; wgt <- row$w
      c <- (1 - d_of(t)) * wgt
      dg[gi_idx(gkey_of(t))] <- dg[gi_idx(gkey_of(t))] + c
      dh[hi_idx(t$S1)] <- dh[hi_idx(t$S1)] + c * gamma
      dh[hi_idx(t$S)] <- dh[hi_idx(t$S)] - c
    }
    for (row in Pc) {
      t <- row$t; wgt <- row$w
      c <- -d_of(t) * wgt
      dg[gi_idx(gkey_of(t))] <- dg[gi_idx(gkey_of(t))] + c
      dh[hi_idx(t$S1)] <- dh[hi_idx(t$S1)] + c * gamma
      dh[hi_idx(t$S)] <- dh[hi_idx(t$S)] - c
    }
    g <- g + lr * (dg - l2 * g)
    h <- h + lr * (dh - l2 * h)
  }

  de <- vapply(Es, function(t) d_of(t), numeric(1))
  dp <- vapply(Ps, function(t) d_of(t), numeric(1))
  # line 6: r = log D - log(1 - D), equivalent to f - log pi.
  reward <- .airl_log(dp) - .airl_log(1 - dp)
  ll <- mean(.airl_log(de)) + mean(.airl_log(1 - dp))
  acc <- (sum(de > 0.5) + sum(dp <= 0.5)) / (length(de) + length(dp))

  g_map <- list()
  for (k in gkeys) g_map[[length(g_map) + 1L]] <-
    setNames(list(g[gi_idx(k)]), deparse(k, control = "useSource"))
  h_map <- list()
  for (k in all_states) h_map[[length(h_map) + 1L]] <-
    setNames(list(h[hi_idx(k)]), deparse(k, control = "useSource"))

  list(estimate = reward, reward = reward, g = g_map, h = h_map,
       f_policy = vapply(Ps, f_of, numeric(1)),
       f_expert = vapply(Es, f_of, numeric(1)),
       D_policy = dp, D_expert = de, accuracy = as.numeric(acc),
       log_likelihood = as.numeric(ll), gamma = gamma,
       state_only = as.logical(state_only),
       method = "AIRL (Fu, Luo & Levine 2018, eq. 4 + Alg. 1)")
}

#' .airl_key_from_str
#'
#' A step of the airl_native implementation. Called by \code{airl}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param s Character; passed to \code{grepl}.
#' @return One of two values, depending on the branch taken.
#' @export
.airl_key_from_str <- function(s) {
  if (grepl("^list\\(", s)) {
    inner <- sub("^list\\((.*)\\)$", "\\1", s)
    parts <- strsplit(inner, ", ", fixed = TRUE)[[1]]
    as.numeric(parts)
  } else {
    as.numeric(s)
  }
}

# local `x %||% y` for the environments above
`%||%` <- function(a, b) if (is.null(a)) b else a

#' Soft value iteration on a deterministic tabular MDP
#'
#' The maximum-entropy :math:`V^*` of eq. Q(s, a) = r(s) + gamma V(s'),
#' V(s) = log sum_a exp Q(s, a). Provided so callers can validate
#' AIRL's recovered reward independently of the fit.
#' @export
soft_value_iteration <- function(states, actions, step, reward, gamma = 0.9,
                                 iters = 2000L, tol = 1e-14) {
  S <- as.list(states); A <- as.list(actions)
  V <- new.env(hash = TRUE, parent = emptyenv())
  for (s in S) assign(deparse(s, control = "useSource"), 0, envir = V)
  for (it in seq_len(as.integer(iters))) {
    newV <- new.env(hash = TRUE, parent = emptyenv())
    for (s in S) {
      qs <- vapply(A, function(a) reward(s) + gamma *
                     get(deparse(step(s, a), control = "useSource"),
                         envir = V), numeric(1))
      m <- max(qs)
      assign(deparse(s, control = "useSource"),
             m + log(sum(exp(qs - m))), envir = newV)
    }
    delta <- max(vapply(S, function(s)
      abs(get(deparse(s, control = "useSource"), envir = newV) -
            get(deparse(s, control = "useSource"), envir = V)), numeric(1)))
    V <- newV
    if (delta < tol) break
  }
  pi <- list()
  for (s in S) {
    qs <- vapply(A, function(a) reward(s) + gamma *
                   get(deparse(step(s, a), control = "useSource"),
                       envir = V), numeric(1))
    Vs <- get(deparse(s, control = "useSource"), envir = V)
    for (i in seq_along(A))
      pi[[length(pi) + 1L]] <-
        setNames(list(exp(qs[i] - Vs)),
                 paste0(deparse(s, control = "useSource"), "|",
                        deparse(A[[i]], control = "useSource")))
  }
  V_out <- list()
  for (s in S) V_out[[length(V_out) + 1L]] <-
    setNames(list(get(deparse(s, control = "useSource"), envir = V)),
             deparse(s, control = "useSource"))
  list(V = V_out, pi = pi)
}

# house entry point: the package exports one morie_<module>
morie_airl <- airl
