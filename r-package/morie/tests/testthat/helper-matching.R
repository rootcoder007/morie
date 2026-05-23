# SPDX-License-Identifier: AGPL-3.0-or-later
# Shared matching-test synthetic-data generator (extracted from
# test-matching.R so any matching test can re-use it).

# Treatment d ~ logit(0.4 x1 + 0.3 x2). Outcome y = tau*d + 0.5 x1 +
# 0.3 x2 + N(0, 0.5). Includes a `region` factor + a `year` numeric
# for tests that exercise exact / CEM matching.
make_match_df <- function(n = 300, tau = 0.4, seed = 1) {
  set.seed(seed)
  x1 <- stats::rnorm(n)
  x2 <- stats::rnorm(n)
  d  <- stats::rbinom(n, 1, stats::plogis(0.4 * x1 + 0.3 * x2))
  y  <- tau * d + 0.5 * x1 + 0.3 * x2 + stats::rnorm(n, sd = 0.5)
  data.frame(d = d, y = y, x1 = x1, x2 = x2,
             region = sample(c("A", "B", "C"), n, replace = TRUE),
             year   = sample(2018:2020, n, replace = TRUE))
}
