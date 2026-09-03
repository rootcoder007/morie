# nashq -- Nash Q-learning for general-sum stochastic games
# Reference: Hu & Wellman (2003) JMLR 4, 1039-1069
# Base R only.

nashq_selections <- c("global_optimal", "saddle", "first", "best_for_agent")

#' nashq_mat
#'
#' A step of the nashq_native implementation. Called by \code{nashq_equilibria_bimatrix},
#' \code{nashq_stage_game_type}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param M A matrix; passed to \code{nrow}.
#' @param name Passed to \code{sprintf}.
#' @return The value of \code{M}, as built in the body.
#' @export
nashq_mat <- function(M, name) {
  M <- as.matrix(M)
  if (nrow(M) == 0L || ncol(M) == 0L) {
    stop(sprintf("nashq: %s must be a non-empty matrix", name))
  }
  storage.mode(M) <- "double"
  M
}

#' nashq_solve
#'
#' A step of the nashq_native implementation. Called by \code{nashq_indifference}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A Passed to \code{cbind}.
#' @param b A vector; its length is taken.
#' @return The value of \code{[}.
#' @export
nashq_solve <- function(A, b) {
  n <- length(b)
  M <- cbind(A, b)
  for (c in seq_len(n)) {
    p <- which.max(abs(M[c:n, c]))
    if (abs(M[p + c - 1L, c]) < 1e-12) return(NULL)
    if (p + c - 1L != c) {
      tmp <- M[c, ]
      M[c, ] <- M[p + c - 1L, ]
      M[p + c - 1L, ] <- tmp
    }
    pv <- M[c, c]
    M[c, ] <- M[c, ] / pv
    for (r in seq_len(n)) {
      if (r == c) next
      f <- M[r, c]
      if (f == 0) next
      M[r, ] <- M[r, ] - f * M[c, ]
    }
  }
  M[, n + 1L]
}

#' nashq_indifference
#'
#' A step of the nashq_native implementation. Called by \code{nashq_equilibria_bimatrix}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param payoff A matrix; indexed by row and column.
#' @param k A count; the body uses it as \code{seq_len(...)}.
#' @return The value of \code{[}.
#' @export
nashq_indifference <- function(payoff, k) {
  Aeq <- matrix(0, nrow = k + 1L, ncol = k + 1L)
  beq <- numeric(k + 1L)
  for (i in seq_len(k)) {
    for (a in seq_len(k)) Aeq[i, a] <- payoff[i, a]
    Aeq[i, k + 1L] <- -1
  }
  Aeq[k + 1L, seq_len(k)] <- 1
  beq[k + 1L] <- 1
  sol <- nashq_solve(Aeq, beq)
  if (is.null(sol)) return(NULL)
  sol[seq_len(k)]
}

#' nashq_payoff
#'
#' A step of the nashq_native implementation. Called by \code{nashq_is_equilibrium},
#' \code{nashq_is_saddle}, \code{nashq_run} and 2 others in the module.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param M A matrix; indexed by row and column.
#' @param p A vector; its length is taken and its elements indexed.
#' @param q A vector; its length is taken and its elements indexed.
#' @return The value of \code{tot}, as built in the body.
#' @export
nashq_payoff <- function(M, p, q) {
  tot <- 0
  for (i in seq_along(p)) {
    if (p[i] == 0) next
    for (j in seq_along(q)) {
      if (q[j] != 0) tot <- tot + p[i] * q[j] * M[i, j]
    }
  }
  tot
}

#' nashq_is_equilibrium
#'
#' A step of the nashq_native implementation. Called by \code{nashq_equilibria_bimatrix}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; indexed by row and column.
#' @param B A matrix; indexed by row and column.
#' @param p Numeric; combined arithmetically in the body.
#' @param q Numeric; combined arithmetically in the body.
#' @param tol Numeric; combined arithmetically in the body.
#' @return A logical value.
#' @export
nashq_is_equilibrium <- function(A, B, p, q, tol) {
  va <- nashq_payoff(A, p, q)
  vb <- nashq_payoff(B, p, q)
  for (i in seq_len(nrow(A))) {
    dev <- sum(q * A[i, ])
    if (dev > va + tol) return(FALSE)
  }
  for (j in seq_len(ncol(A))) {
    dev <- sum(p * B[, j])
    if (dev > vb + tol) return(FALSE)
  }
  TRUE
}

#' nashq_equilibria_bimatrix
#'
#' A step of the nashq_native implementation. Called by \code{nashq_select_eq},
#' \code{nashq_stage_game_type}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; indexed by row and column.
#' @param B A matrix; indexed by row and column.
#' @param tol Numeric; combined arithmetically in the body. Defaults to \code{1e-09}.
#' @return The value of \code{out}, as built in the body.
#' @export
nashq_equilibria_bimatrix <- function(A, B, tol = 1e-9) {
  A <- nashq_mat(A, "A")
  B <- nashq_mat(B, "B")
  if (nrow(B) != nrow(A) || ncol(B) != ncol(A)) {
    stop("nashq: A and B must have the same shape")
  }
  m <- nrow(A)
  n <- ncol(A)
  out <- list()
  seen <- list()
  for (k in seq_len(min(m, n))) {
    I_all <- combn(m, k)
    J_all <- combn(n, k)
    for (ii in seq_len(ncol(I_all))) {
      I <- I_all[, ii]
      for (jj in seq_len(ncol(J_all))) {
        J <- J_all[, jj]
        subA <- A[I, J, drop = FALSE]
        subB <- t(B[I, J, drop = FALSE])
        q <- nashq_indifference(subA, k)
        p <- nashq_indifference(subB, k)
        if (is.null(q) || is.null(p)) next
        if (min(q) < -tol || min(p) < -tol) next
        P <- numeric(m)
        Q <- numeric(n)
        for (a in seq_along(I)) P[I[a]] <- max(0, p[a])
        for (a in seq_along(J)) Q[J[a]] <- max(0, q[a])
        sp <- sum(P)
        sq <- sum(Q)
        if (sp <= tol || sq <= tol) next
        P <- P / sp
        Q <- Q / sq
        if (!nashq_is_equilibrium(A, B, P, Q, tol)) next
        key <- paste(paste(round(P, 9), collapse = ","),
                     paste(round(Q, 9), collapse = ","), sep = "|")
        if (!is.null(seen[[key]])) next
        seen[[key]] <- TRUE
        out[[length(out) + 1L]] <- list(p = P, q = Q)
      }
    }
  }
  out
}

#' nashq_is_saddle
#'
#' A step of the nashq_native implementation. Called by \code{nashq_select_eq},
#' \code{nashq_stage_game_type}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; passed to \code{nrow}.
#' @param B Passed to \code{nashq_payoff}.
#' @param p Passed to \code{nashq_payoff}.
#' @param q Passed to \code{nashq_payoff}.
#' @param tol Numeric; combined arithmetically in the body.
#' @return A logical value.
#' @export
nashq_is_saddle <- function(A, B, p, q, tol) {
  for (j in seq_len(ncol(A))) {
    pure <- numeric(ncol(A))
    pure[j] <- 1
    if (nashq_payoff(A, p, pure) < nashq_payoff(A, p, q) - tol) return(FALSE)
  }
  for (i in seq_len(nrow(A))) {
    pure <- numeric(nrow(A))
    pure[i] <- 1
    if (nashq_payoff(B, pure, q) < nashq_payoff(B, p, q) - tol) return(FALSE)
  }
  TRUE
}

#' nashq_stage_game_type
#'
#' A step of the nashq_native implementation. Called by \code{nashq_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A Numeric; passed to \code{max}.
#' @param B Numeric; passed to \code{max}.
#' @param tol Numeric; combined arithmetically in the body. Defaults to \code{1e-09}.
#' @return A list with \code{estimate}, \code{equilibria}, \code{n_equilibria},
#' \code{has_global_optimal}, \code{has_saddle}, \code{global_optimal}, \code{saddle},
#' \code{method}.
#' @export
nashq_stage_game_type <- function(A, B, tol = 1e-9) {
  A <- nashq_mat(A, "A")
  B <- nashq_mat(B, "B")
  eqs <- nashq_equilibria_bimatrix(A, B, tol)
  best_a <- max(A)
  best_b <- max(B)
  glob <- list()
  sad <- list()
  for (eq in eqs) {
    p <- eq$p
    q <- eq$q
    va <- nashq_payoff(A, p, q)
    vb <- nashq_payoff(B, p, q)
    if (va >= best_a - tol && vb >= best_b - tol) glob[[length(glob) + 1L]] <- eq
    if (nashq_is_saddle(A, B, p, q, tol)) sad[[length(sad) + 1L]] <- eq
  }
  list(estimate = length(eqs), equilibria = eqs, n_equilibria = length(eqs),
       has_global_optimal = length(glob) > 0L, has_saddle = length(sad) > 0L,
       global_optimal = glob, saddle = sad,
       method = "stage game classification (Hu & Wellman 2003 Defs 12-13)")
}

#' nashq_select_eq
#'
#' A step of the nashq_native implementation. Called by \code{nashq_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A Numeric; passed to \code{max}.
#' @param B Numeric; passed to \code{max}.
#' @param selection One of \code{"best_for_agent"}, \code{"first"}, \code{"global_optimal"}.
#' @param agent Passed to \code{==}.
#' @param tol Numeric; combined arithmetically in the body.
#' @return The value of \code{[[}.
#' @export
nashq_select_eq <- function(A, B, selection, agent, tol) {
  eqs <- nashq_equilibria_bimatrix(A, B, tol)
  if (length(eqs) == 0L) return(NULL)
  if (selection == "first") return(eqs[[1]])
  if (selection == "best_for_agent") {
    M <- if (agent == 0L) A else B
    return(eqs[[which.max(sapply(eqs, function(e) nashq_payoff(M, e$p, e$q)))]])
  }
  if (selection == "global_optimal") {
    ba <- max(A)
    bb <- max(B)
    for (eq in eqs) {
      if (nashq_payoff(A, eq$p, eq$q) >= ba - tol &&
          nashq_payoff(B, eq$p, eq$q) >= bb - tol) return(eq)
    }
    return(eqs[[1]])
  }
  for (eq in eqs) if (nashq_is_saddle(A, B, eq$p, eq$q, tol)) return(eq)
  eqs[[1]]
}

#' nashq_pick
#'
#' A step of the nashq_native implementation. Called by \code{nashq_run}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param M Passed to \code{rowMeans}.
#' @param A A vector; its length is taken.
#' @param who Passed to \code{==}.
#' @param epsilon Passed to \code{<}.
#' @param rng Accepted by the signature and not used anywhere in the body.
#' @return One of two values, depending on the branch taken.
#' @export
nashq_pick <- function(M, A, who, epsilon, rng) {
  if (rng() < epsilon) return(sample.int(length(A), 1L) - 1L)
  if (who == 0L) {
    vals <- rowMeans(M)
  } else {
    vals <- colMeans(M)
  }
  bv <- max(vals)
  best <- which(vals >= bv - 1e-15)
  pick <- if (length(best) > 1L) best[as.integer(rng() * length(best)) + 1L] else best
  pick - 1L
}

#' nashq_run
#'
#' A step of the nashq_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param states Coerced to list by the body, with \code{as.list}.
#' @param actions A vector; its length is taken and its elements indexed.
#' @param step A function; the body checks with \code{is.function}.
#' @param rewards A function; the body checks with \code{is.function}.
#' @param gamma Numeric; combined arithmetically in the body. Defaults to \code{0.9}.
#' @param alpha Numeric; combined arithmetically in the body. Defaults to \code{0.5}.
#' @param epsilon Passed to \code{nashq_pick}. Defaults to \code{0.1}.
#' @param episodes Coerced to integer by the body, with \code{as.integer}. Defaults to \code{500}.
#' @param horizon Coerced to integer by the body, with \code{as.integer}. Defaults to \code{50}.
#' @param start Optional; may be \code{NULL}. A function; the body checks with \code{is.function}.
#' @param selection Carried through into a list the body builds. Defaults to
#' \code{"global_optimal"}.
#' @param terminal Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.
#' @param seed Passed to \code{set.seed}. Defaults to \code{0}.
#' @param agent Passed to \code{nashq_select_eq}. Defaults to \code{0}.
#' @param tol Passed to \code{nashq_select_eq}. Defaults to \code{1e-09}.
#' @return A list with \code{estimate}, \code{q}, \code{policy}, \code{nash_values},
#' \code{stage_game_types}, \code{returns}, \code{mean_return_last}, \code{selection},
#' \code{method}.
#' @export
nashq_run <- function(states, actions, step, rewards, gamma = 0.9, alpha = 0.5,
                      epsilon = 0.1, episodes = 500, horizon = 50, start = NULL,
                      selection = "global_optimal", terminal = NULL,
                      seed = 0, agent = 0, tol = 1e-9) {
  if (!(selection %in% nashq_selections)) {
    stop(sprintf("nashq: selection must be one of %s, got %s",
                 paste(nashq_selections, collapse = ", "), selection))
  }
  S <- as.list(states)
  if (length(actions) != 2L) {
    stop("nashq: this implementation covers two players; pass actions as (A1, A2)")
  }
  A1 <- as.list(actions[[1]])
  A2 <- as.list(actions[[2]])
  if (length(S) == 0L || length(A1) == 0L || length(A2) == 0L) {
    stop("nashq: states and both action sets must be non-empty")
  }
  if (!is.function(step) || !is.function(rewards)) {
    stop("nashq: step and rewards must be callable")
  }
  term <- if (is.null(terminal)) list() else as.list(terminal)
  s0 <- if (is.function(start)) start else (function() if (is.null(start)) S[[1]] else start)
  set.seed(seed)
  rng <- function() runif(1)
  Q <- list()
  for (pl in 0:1) {
    for (s in S) {
      Q[[paste(pl, as.character(s), sep = "|")]] <- matrix(0, nrow = length(A1), ncol = length(A2))
    }
  }
  returns <- list()
  q_get <- function(pl, s) Q[[paste(pl, as.character(s), sep = "|")]]
  q_set <- function(pl, s, M) Q[[paste(pl, as.character(s), sep = "|")]] <<- M
  for (ep in seq_len(as.integer(episodes))) {
    s <- s0()
    tot <- c(0, 0)
    for (t in seq_len(as.integer(horizon))) {
      if (s %in% term) break
      M0 <- q_get(0L, s)
      M1 <- q_get(1L, s)
      i <- nashq_pick(M0, A1, 0L, epsilon, rng) + 1L
      j <- nashq_pick(M1, A2, 1L, epsilon, rng) + 1L
      s1 <- step(s, A1[[i]], A2[[j]])
      r1 <- 0
      r2 <- 0
      rr <- rewards(s, A1[[i]], A2[[j]], s1)
      r1 <- as.numeric(rr[[1]])
      r2 <- as.numeric(rr[[2]])
      tot[1L] <- tot[1L] + r1
      tot[2L] <- tot[2L] + r2
      if (s1 %in% term) {
        nv <- c(0, 0)
      } else {
        eq <- nashq_select_eq(q_get(0L, s1), q_get(1L, s1), selection, agent, tol)
        if (is.null(eq)) {
          nv <- c(0, 0)
        } else {
          nv <- c(nashq_payoff(q_get(0L, s1), eq$p, eq$q),
                  nashq_payoff(q_get(1L, s1), eq$p, eq$q))
        }
      }
      for (pair in list(c(0L, r1), c(1L, r2))) {
        pl <- pair[1]
        r <- pair[2]
        cur <- q_get(pl, s)[i, j]
        q_set(pl, s, q_get(pl, s))
        M <- q_get(pl, s)
        M[i, j] <- (1 - alpha) * cur + alpha * (r + gamma * nv[pl + 1L])
        q_set(pl, s, M)
      }
      s <- s1
    }
    returns[[length(returns) + 1L]] <- tot
  }
  policy <- list()
  nash_values <- list()
  types <- list()
  for (s in S) {
    cls <- nashq_stage_game_type(q_get(0L, s), q_get(1L, s), tol)
    types[[as.character(s)]] <- if (cls$has_global_optimal) "global_optimal"
      else if (cls$has_saddle) "saddle"
      else if (cls$n_equilibria > 0L) "neither" else "none_found"
    eq <- nashq_select_eq(q_get(0L, s), q_get(1L, s), selection, agent, tol)
    if (is.null(eq)) next
    policy[[as.character(s)]] <- list(p = eq$p, q = eq$q)
    nash_values[[as.character(s)]] <- c(nashq_payoff(q_get(0L, s), eq$p, eq$q),
                                        nashq_payoff(q_get(1L, s), eq$p, eq$q))
  }
  tenth <- max(1L, as.integer(episodes) %/% 10L)
  last10 <- tail(returns, tenth)
  mean_last <- c(sum(vapply(last10, function(r) r[1L], numeric(1))) / tenth,
                 sum(vapply(last10, function(r) r[2L], numeric(1))) / tenth)
  list(estimate = Q, q = Q, policy = policy, nash_values = nash_values,
       stage_game_types = types, returns = returns,
       mean_return_last = mean_last, selection = selection,
       method = "Nash Q-learning (Hu & Wellman 2003, Table 2)")
}

#' nashq_cheatsheet
#'
#' A step of the nashq_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
nashq_cheatsheet <- function() {
  paste("nashq: Q^i over JOINT actions; update with the stage-game Nash payoff instead of a max -- Q^i <- (1-a)Q^i + a[r^i + beta pi^1...pi^n Q^i(s')] (Hu & Wellman 2003 eqs. 6-7). Needs every agent's reward. Equilibrium selection changes the update: convergence is proved only for global optimal (Def 12) or saddle (Def 13) stage games. stage_game_type() reports which you have.")
}

# house entry point: the package exports one morie_<module>
morie_nashq <- nashq_mat
