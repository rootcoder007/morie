# miprgr -- branch and bound: solve the relaxation, then split on a fraction
# References:
#   Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
#   Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
# Base R only.

#' Two-phase simplex with Bland\'s rule, maximising c\'x
#'
#' A step of the miprgr_native implementation. Called by \code{miprgr_solve_relaxation}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; indexed by row and column.
#' @param b A vector; its length is taken and its elements indexed.
#' @param c A vector; its length is taken.
#' @param tol Numeric; combined arithmetically in the body. Defaults to \code{1e-09}.
#' @param max_iter Defaults to \code{20000}.
#' @return A list with \code{feasible}, \code{x}, \code{value}.
#' @export
miprgr_simplex <- function(A, b, c, tol = 1e-9, max_iter = 20000) {
  # Two-phase simplex with Bland's rule, maximising c'x.
  m <- length(b)
  n <- length(c)
  if (m == 0L) {
    return(list(feasible = TRUE, x = rep(0, n), value = 0))
  }
  rows <- list()
  rhs <- numeric(m)
  for (i in seq_len(m)) {
    rows[[i]] <- as.numeric(A[i, ])
    rhs[i] <- as.numeric(b[i])
    if (rhs[i] < 0) {
      rows[[i]] <- -rows[[i]]
      rhs[i] <- -rhs[i]
    }
  }
  width <- n + m
  Tmat <- matrix(0, nrow = m, ncol = width + 1L)
  for (i in seq_len(m)) {
    Tmat[i, seq_len(n)] <- rows[[i]]
    Tmat[i, n + i] <- 1
    Tmat[i, width + 1L] <- rhs[i]
  }
  basis <- as.integer(n + seq_len(m))
  sign_neg <- rep(1, m)
  sign_neg[which(b < 0)] <- -1
  Tmat[, seq_len(n)] <- Tmat[, seq_len(n)] * sign_neg
  need_art <- which(b < 0)
  na <- length(need_art)
  width2 <- n + m + na
  T2 <- matrix(0, nrow = m, ncol = width2 + 1L)
  for (i in seq_len(m)) {
    T2[i, seq_len(n)] <- rows[[i]]
    T2[i, n + i] <- 1
    T2[i, width2 + 1L] <- rhs[i]
  }
  basis2 <- as.integer(n + seq_len(m))
  if (na > 0L) {
    for (a in seq_along(need_art)) {
      T2[need_art[a], n + m + a] <- 1
      basis2[need_art[a]] <- n + m + a
    }
  }

  reduced <- function(obj) {
    z <- obj
    for (i in seq_len(m)) {
      f <- z[basis2[i] + 1L]
      if (f != 0) z <- z - f * T2[i, ]
    }
    z
  }

  pivot <- function(pr, pc) {
    pv <- T2[pr, pc]
    T2[pr, ] <- T2[pr, ] / pv
    for (i in seq_len(m)) {
      if (i != pr && T2[i, pc] != 0) {
        f <- T2[i, pc]
        T2[i, ] <- T2[i, ] - f * T2[pr, ]
      }
    }
    basis2[pr] <<- pc - 1L
  }

  run_phase <- function(obj, allowed, maxit) {
    for (iter in seq_len(maxit)) {
      z <- reduced(obj)
      enter <- NA_integer_
      for (j in allowed) {
        if (z[j] < -tol) { enter <- j; break }
      }
      if (is.na(enter)) return(TRUE)
      ratio <- Inf
      leave <- NA_integer_
      for (i in seq_len(m)) {
        if (T2[i, enter] > tol) {
          r <- T2[i, width2 + 1L] / T2[i, enter]
          cand_basis <- basis2[i] + 1L
          if (r < ratio - tol ||
              (abs(r - ratio) <= tol && !is.na(leave) &&
               cand_basis < basis2[leave] + 1L)) {
            ratio <- r
            leave <- i
          }
        }
      }
      if (is.na(leave)) return(FALSE)
      pivot(leave, enter)
    }
    FALSE
  }

  if (na > 0L) {
    phase1 <- rep(0, width2 + 1L)
    for (a in seq_len(na)) phase1[n + m + a] <- 1
    if (!run_phase(phase1, seq_len(n + m), max_iter)) {
      return(list(feasible = FALSE, x = NULL, value = NULL))
    }
    infeas <- 0
    for (i in seq_len(m)) {
      if (basis2[i] + 1L >= n + m + 1L) infeas <- infeas + T2[i, width2 + 1L]
    }
    if (infeas > 1e-7) {
      return(list(feasible = FALSE, x = NULL, value = NULL))
    }
    for (i in seq_len(m)) {
      if (basis2[i] + 1L >= n + m + 1L) {
        for (j in seq_len(n + m)) {
          if (abs(T2[i, j]) > tol) {
            pivot(i, j)
            break
          }
        }
      }
    }
  }
  phase2 <- rep(0, width2 + 1L)
  phase2[seq_len(n)] <- -as.numeric(c)
  if (!run_phase(phase2, seq_len(n + m), max_iter)) {
    return(list(feasible = FALSE, x = NULL, value = NULL))
  }
  x <- rep(0, n)
  for (i in seq_len(m)) {
    if (basis2[i] + 1L <= n) x[basis2[i] + 1L] <- T2[i, width2 + 1L]
  }
  list(feasible = TRUE, x = x, value = sum(as.numeric(c) * x))
}

#' miprgr_solve_relaxation
#'
#' A step of the miprgr_native implementation. Called by \code{miprgr_branch_and_bound}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; indexed by row and column.
#' @param b A vector; its length is taken and its elements indexed.
#' @param c A vector; its length is taken.
#' @param bounds A vector; its length is taken. Defaults to \code{list()}.
#' @param n Optional; may be \code{NULL}. Coerced to integer by the body, with \code{as.integer}.
#' @param maximise A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param solver Compared against \code{"simplex"}. Defaults to \code{"simplex"}.
#' @return A list with \code{feasible}, \code{x}, \code{value}, \code{note}.
#' @export
miprgr_solve_relaxation <- function(A, b, c, bounds = list(),
                                    n = NULL, maximise = TRUE,
                                    solver = "simplex") {
  nn <- if (is.null(n)) length(c) else as.integer(n)
  if (solver != "simplex") {
    stop("miprgr: only simplex solver is implemented in base R here")
  }
  rows <- list()
  rhs <- numeric(length(b) + length(bounds))
  idx <- 0L
  for (i in seq_along(b)) {
    idx <- idx + 1L
    rows[[idx]] <- as.numeric(A[i, ])
    rhs[idx] <- as.numeric(b[i])
  }
  for (e in bounds) {
    idx <- idx + 1L
    r <- rep(0, nn)
    j <- as.integer(e$var)
    if (e$sense == "le") {
      r[j] <- 1
      rhs[idx] <- as.numeric(e$value)
    } else {
      r[j] <- -1
      rhs[idx] <- -as.numeric(e$value)
    }
    rows[[idx]] <- r
  }
  M <- do.call(rbind, rows)
  sgn <- if (maximise) 1 else -1
  out <- miprgr_simplex(M, rhs, sgn * as.numeric(c))
  if (!out$feasible) {
    return(list(feasible = FALSE, x = NULL, value = NULL,
                note = "the relaxation is infeasible, so every integer point below this node is too"))
  }
  x <- pmax(0, out$x)
  val <- sum(as.numeric(c) * x)
  list(feasible = TRUE, x = x, value = val,
       note = "a valid BOUND on every integer point below this node")
}

#' miprgr_fractional_variable
#'
#' A step of the miprgr_native implementation. Called by \code{miprgr_branch_and_bound}.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x A vector; indexed elementwise.
#' @param integer_vars See Usage.
#' @param tol Defaults to \code{1e-07}.
#' @return A list with \code{index}, \code{fractionality}, \code{integral}.
#' @export
miprgr_fractional_variable <- function(x, integer_vars, tol = 1e-7) {
  best <- NA_integer_
  gap <- 0
  for (j in integer_vars) {
    v <- as.numeric(x[j])
    f <- abs(v - round(v))
    if (f > tol && f > gap) {
      best <- as.integer(j)
      gap <- f
    }
  }
  list(index = best, fractionality = gap,
       integral = is.na(best))
}

#' miprgr_round_relaxation
#'
#' A step of the miprgr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param x Coerced to numeric by the body, with \code{as.numeric}.
#' @param A A matrix; indexed by row and column.
#' @param b A vector; indexed elementwise.
#' @param integer_vars See Usage.
#' @return A list with \code{x}, \code{feasible}, \code{violations}, \code{note}.
#' @export
miprgr_round_relaxation <- function(x, A, b, integer_vars) {
  xr <- as.numeric(x)
  for (j in integer_vars) xr[j] <- as.numeric(round(xr[j]))
  viol <- list()
  for (i in seq_len(nrow(A))) {
    lhs <- sum(as.numeric(A[i, ]) * xr)
    if (lhs > as.numeric(b[i]) + 1e-7) {
      viol[[length(viol) + 1L]] <- list(row = i, lhs = lhs, rhs = as.numeric(b[i]))
    }
  }
  list(x = xr, feasible = length(viol) == 0L, violations = viol,
       note = "rounding is not a substitute for branching")
}

#' miprgr_enumerate_integer
#'
#' A step of the miprgr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A A matrix; indexed by row and column.
#' @param b A vector; indexed elementwise.
#' @param c A vector; its length is taken.
#' @param integer_vars Coerced to integer by the body, with \code{as.integer}.
#' @param upper Coerced to integer by the body, with \code{as.integer}. Defaults to \code{10}.
#' @param maximise A flag; the body branches on it. Defaults to \code{TRUE}.
#' @return A list with \code{value}, \code{x}, \code{note}.
#' @export
miprgr_enumerate_integer <- function(A, b, c, integer_vars, upper = 10,
                                     maximise = TRUE) {
  n <- length(c)
  integer_vars <- as.integer(integer_vars)
  best <- if (maximise) -Inf else Inf
  best_x <- NULL
  stack <- list(integer(0))
  while (length(stack) > 0L) {
    pre <- stack[[length(stack)]]
    stack[[length(stack)]] <- NULL
    if (length(pre) == n) {
      ok <- TRUE
      for (i in seq_len(nrow(A))) {
        lhs <- sum(as.numeric(A[i, ]) * pre)
        if (lhs > as.numeric(b[i]) + 1e-7) { ok <- FALSE; break }
      }
      if (ok) {
        val <- sum(as.numeric(c) * pre)
        better <- if (maximise) val > best else val < best
        if (better) {
          best <- val
          best_x <- as.numeric(pre)
        }
      }
      next
    }
    j <- length(pre) + 1L
    pre_num <- as.numeric(pre)
    for (v in seq_len(as.integer(upper) + 1L) - 1L) {
      np_ <- c(pre_num, as.numeric(v))
      stack[[length(stack) + 1L]] <- np_
    }
  }
  list(value = best, x = best_x,
       note = "exhaustive over the box, so the search can be checked against something other than itself")
}

#' miprgr_branch_and_bound
#'
#' A step of the miprgr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @param A See Usage.
#' @param b See Usage.
#' @param c A vector; its length is taken.
#' @param integer_vars Coerced to integer by the body, with \code{as.integer}.
#' @param maximise A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param prune A flag; the body branches on it. Defaults to \code{TRUE}.
#' @param max_nodes Coerced to integer by the body, with \code{as.integer}. Defaults to \code{5000}.
#' @param solver Defaults to \code{"simplex"}.
#' @return A list with \code{estimate}, \code{value}, \code{x}, \code{feasible}, \code{nodes}, \code{pruned}, \code{pruning}, \code{max_list_length}, \code{root_bound}, \code{truncated}, \code{method}, \code{note}.
#' @export
miprgr_branch_and_bound <- function(A, b, c, integer_vars, maximise = TRUE,
                                    prune = TRUE, max_nodes = 5000,
                                    solver = "simplex") {
  n <- length(c)
  I <- sort(unique(as.integer(integer_vars)))
  if (any(I < 1L | I > n)) {
    stop("miprgr: an integer index is outside the variable set")
  }
  if (maximise) {
    better <- function(a, bb) a > bb + 1e-7
  } else {
    better <- function(a, bb) a < bb - 1e-7
  }
  incumbent <- if (maximise) -Inf else Inf
  inc_x <- NULL
  lst <- list()
  nodes <- 0L
  pruned <- 0L
  max_len <- 0L
  root_bound <- NULL
  while (nodes < as.integer(max_nodes)) {
    bounds <- lapply(lst, function(e) list(var = e$var, sense = e$sense, value = e$value))
    rel <- miprgr_solve_relaxation(A, b, c, bounds, n, maximise, solver)
    nodes <- nodes + 1L
    max_len <- max(max_len, length(lst))
    if (is.null(root_bound) && isTRUE(rel$feasible)) root_bound <- rel$value
    descend <- FALSE
    if (isTRUE(rel$feasible)) {
      cut <- prune && !is.null(inc_x) && !better(rel$value, incumbent)
      if (cut) {
        pruned <- pruned + 1L
      } else {
        fv <- miprgr_fractional_variable(rel$x, I)
        if (isTRUE(fv$integral)) {
          if (better(rel$value, incumbent)) {
            incumbent <- rel$value
            inc_x <- sapply(seq_len(n), function(j) {
              if (j %in% I) as.numeric(round(rel$x[j])) else as.numeric(rel$x[j])
            })
          }
        } else {
          j <- fv$index
          v <- rel$x[j]
          lst[[length(lst) + 1L]] <- list(
            var = j, sense = "le", value = floor(v),
            alt = c("ge", ceiling(v)), marked = FALSE)
          descend <- TRUE
        }
      }
    }
    if (descend) next
    repeat {
      if (length(lst) == 0L) {
        return(list(
          estimate = if (!is.null(inc_x)) incumbent else NULL,
          value = if (!is.null(inc_x)) incumbent else NULL,
          x = inc_x, feasible = !is.null(inc_x),
          nodes = nodes, pruned = pruned, pruning = prune,
          max_list_length = max_len, root_bound = root_bound,
          method = "branch and bound; Land & Doig (1960), Dakin (1965) Fig. 2",
          note = "the list holds the current PATH, so its length is the tree depth, not the number of open nodes"))
      }
      last <- lst[[length(lst)]]
      if (isTRUE(last$marked)) {
        lst[[length(lst)]] <- NULL
        next
      }
      sense_alt <- last$alt[1]
      val_alt <- as.numeric(last$alt[2])
      last$alt <- c(last$sense, last$value)
      last$sense <- sense_alt
      last$value <- val_alt
      last$marked <- TRUE
      lst[[length(lst)]] <- last
      break
    }
  }
  list(
    estimate = if (!is.null(inc_x)) incumbent else NULL,
    value = if (!is.null(inc_x)) incumbent else NULL,
    x = inc_x, feasible = !is.null(inc_x), nodes = nodes,
    pruned = pruned, pruning = prune,
    max_list_length = max_len, root_bound = root_bound,
    truncated = TRUE,
    method = "branch and bound; Land & Doig (1960), Dakin (1965) Fig. 2",
    note = "node limit reached, so the result is NOT proven optimal")
}

#' miprgr_cheatsheet
#'
#' A step of the miprgr_native implementation. No other function in the package calls it.
#' See the file header for the source the module follows.
#' source it follows.
#'
#' @return A character value.
#' @export
miprgr_cheatsheet <- function() {
  paste("miprgr: the LP relaxation is easy and usually FRACTIONAL,",
        "and ROUNDING is not a fix -- the rounded point is often",
        "infeasible and, when feasible, often strictly worse.",
        "BOUND: the relaxation's value bounds every integer point",
        "below that node, so once an incumbent exists any node no",
        "better than it can be discarded UNEXPLORED. BRANCH: Land",
        "and Doig enumerated values; DAKIN's dichotomy x_j <=",
        "floor(v) OR x_j >= ceil(v) excludes the fractional point,",
        "keeps every integer point, and leaves each node an LP --",
        "so the whole search is a binary tree of LPs. Pruning must",
        "not change the optimum: run without it and compare.")
}

# house entry point: the package exports one morie_<module>
morie_miprgr <- miprgr_simplex
