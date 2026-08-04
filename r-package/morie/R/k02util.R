# SPDX-License-Identifier: AGPL-3.0-or-later
# k02 batch shared helpers -- internal, not exported.
# Mirrors src/morie/fn/k02util.py arithmetic exactly.

k02fe <- function(y, v) {
  y <- as.numeric(y); v <- as.numeric(v)
  w <- 1 / v
  sw <- sum(w)
  mu <- sum(w * y) / sw
  q <- sum(w * (y - mu)^2)
  list(mu = mu, var = 1 / sw, sw = sw, Q = q, df = length(y) - 1L)
}

k02dl <- function(y, v) {
  y <- as.numeric(y); v <- as.numeric(v)
  fe <- k02fe(y, v)
  w <- 1 / v
  cc <- fe$sw - sum(w * w) / fe$sw
  tau2 <- if (cc > 0) max(0, (fe$Q - fe$df) / cc) else 0
  ws <- 1 / (v + tau2)
  sws <- sum(ws)
  list(tau2 = tau2, mu = sum(ws * y) / sws, var = 1 / sws, Q = fe$Q, df = fe$df)
}

k02mm <- function(y, v, tau0) {
  y <- as.numeric(y); v <- as.numeric(v)
  a <- 1 / (v + tau0)
  sa <- sum(a); sa2 <- sum(a * a)
  yb <- sum(a * y) / sa
  num <- sum(a * (y - yb)^2) - sum(a * v) + sum(a * a * v) / sa
  den <- sa - sa2 / sa
  if (den > 0) max(0, num / den) else 0
}

k02z <- function(p) stats::qnorm(p)
k02tq <- function(p, df) stats::qt(p, df)
k02p2z <- function(z) 2 * stats::pnorm(abs(z), lower.tail = FALSE)
k02p2t <- function(tv, df) 2 * stats::pt(abs(tv), df, lower.tail = FALSE)
k02pchi <- function(q, df) stats::pchisq(q, df, lower.tail = FALSE)
