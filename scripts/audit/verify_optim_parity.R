# SPDX-License-Identifier: AGPL-3.0-or-later
# Base-R parity harness: morie R optimiser functions vs anchors from live Python.
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; anchor_file <- args[2]
source(src)

read_anchor <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "")
  out <- list()
  for (m in regmatches(txt, gregexpr('"[^"]+"\\s*:\\s*(\\[[^]]*\\]|-?[0-9.eE+-]+)', txt))[[1]]) {
    key <- sub('^"([^"]+)".*$', "\\1", m)
    val <- sub('^"[^"]+"\\s*:\\s*', "", m)
    val <- gsub("\\[|\\]", "", val)
    out[[key]] <- as.numeric(strsplit(val, ",")[[1]])
  }
  out
}
A <- read_anchor(anchor_file)

pass <- 0L; fail <- 0L
chk <- function(label, got, want, tol = 1e-10) {
  got <- as.numeric(got); want <- as.numeric(want)
  ok <- length(got) == length(want) && max(abs(got - want)) < tol
  if (ok) pass <<- pass + 1L else {
    fail <<- fail + 1L
    cat(sprintf("  FAIL %-16s got=%s want=%s\n", label,
                paste(signif(got, 10), collapse = ","),
                paste(signif(want, 10), collapse = ",")))
  }
}

r <- morie_adam(c(1, -2, 0.5), lr = 0.01)
chk("adam1", r$update, A$adam1)
chk("adam2", morie_adam(c(1, -2, 0.5), lr = 0.01, state = r$state)$update, A$adam2)
chk("adagrad", morie_adagrad(c(1, 2), lr = 0.1)$update, A$adagrad)
chk("rmsprop", morie_rmsprop(c(1, 2), lr = 0.1)$update, A$rmsprop)
chk("sgdmom", morie_sgd_momentum(c(1, 2), mu = 0.9, lr = 0.1)$update, A$sgdmom)
chk("nesterov", morie_nesterov(c(1, 2), mu = 0.9, lr = 0.1)$update, A$nesterov)
chk("gdupd", morie_gradient_descent_update(c(1, 2), c(0.5, -0.5), 0.1)$beta, A$gdupd)
sg <- morie_sgd_update(c(0, 0), rbind(c(1, 2), c(3, 4)), eta = 0.1)
chk("sgdup_beta", sg$beta, A$sgdup_beta)
chk("sgdup_se", sg$grad_se, A$sgdup_se)
chk("adamw", morie_adamw_step(c(1, 2), lr = 0.1, wd = 0.5, theta = c(2, -1))$update, A$adamw)
la <- morie_lars(c(1, 1), c(2, 2), lr = 0.1, mu = 0)
chk("lars_upd", la$update, A$lars_upd); chk("lars_trust", la$trust_ratio, A$lars_trust)
lb <- morie_lamb(c(1, 1), c(2, 2), lr = 0.1, wd = 0)
chk("lamb_upd", lb$update, A$lamb_upd); chk("lamb_trust", lb$trust_ratio, A$lamb_trust)
H <- matrix(c(4, 1, 1, 3), 2); xx <- c(5, -2); gg <- as.numeric(H %*% xx)
chk("newton", morie_boyd_newton(gg, H)$step, A$newton)
chk("decrement", morie_boyd_newton_decrement(gg, H)$decrement, A$decrement)
chk("backtrack_t", morie_boyd_backtracking(function(z) sum(z^2), 2, 1, -2)$t, A$backtrack_t)
chk("prox_l1", morie_boyd_proximal("l1", c(3, -0.5, 0.2), t = 1)$prox, A$prox_l1)
chk("prox_l2", morie_boyd_proximal("l2", c(3, 4), t = 1)$prox, A$prox_l2)

cat(sprintf("\noptimiser parity: %d passed, %d failed\n", pass, fail))
if (fail > 0) quit(status = 1)
