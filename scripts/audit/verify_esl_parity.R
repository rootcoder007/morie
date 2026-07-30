# SPDX-License-Identifier: AGPL-3.0-or-later
# Base-R parity harness: morie R ESL functions vs anchors from live Python.
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; D <- args[2]
source(src)

rd <- function(n) as.matrix(utils::read.csv(file.path(D, paste0(n, ".csv")), header = FALSE))
v  <- function(n) as.numeric(rd(n))
read_anchor <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "")
  out <- list()
  for (m in regmatches(txt, gregexpr('"[^"]+"\\s*:\\s*\\[[^]]*\\]', txt))[[1]]) {
    key <- sub('^"([^"]+)".*$', "\\1", m)
    val <- gsub('^"[^"]+"\\s*:\\s*\\[|\\]$', "", m)
    out[[key]] <- as.numeric(strsplit(val, ",")[[1]])
  }
  out
}
A <- read_anchor(file.path(D, "anchors.json"))

pass <- 0L; fail <- 0L
chk <- function(label, got, want, tol = 1e-8) {
  got <- as.numeric(got); want <- as.numeric(want)
  ok <- length(got) == length(want) && max(abs(got - want)) < tol
  if (ok) pass <<- pass + 1L else {
    fail <<- fail + 1L
    cat(sprintf("  FAIL %-14s maxdiff=%.3g\n    got =%s\n    want=%s\n", label,
        if (length(got) == length(want)) max(abs(got - want)) else NA,
        paste(signif(utils::head(got, 6), 10), collapse = ","),
        paste(signif(utils::head(want, 6), 10), collapse = ",")))
  }
}

Xg <- rd("Xg"); yg <- v("yg"); Xs <- rd("Xs"); ys <- v("ys")
noisy <- v("noisy"); xm <- v("xm"); Xcv <- rd("Xcv"); ycv <- v("ycv")

chk("wgt_grad", morie_esl_weight_decay(c(3, -4), lambda_ = 0.5)$gradient, A$wgt_grad)
chk("wgt_pen",  morie_esl_weight_decay(c(3, -4), lambda_ = 0.5)$penalty, A$wgt_pen)
chk("wgt_l1",   morie_esl_weight_decay(c(2, 0, -1), lambda_ = 1, norm = "l1")$gradient, A$wgt_l1)
md <- morie_esl_mdl(-120, 4, n = 100)
chk("mdl", c(md$mdl, md$bic, md$model_cost), A$mdl)

iw <- morie_esl_iwls(Xg, yg)
chk("iwls_beta", iw$beta, A$iwls_beta)
chk("iwls_ll",   iw$loglik, A$iwls_ll)
chk("iwls_se",   iw$se, A$iwls_se)
lg <- morie_esl_logistic_reg(Xg, yg)
chk("lg_or",  lg$odds_ratio, A$lg_or)
chk("lg_acc", lg$accuracy, A$lg_acc)
sp <- morie_esl_iwls(matrix(c(-2, -1, 1, 2), ncol = 1), c(0, 0, 1, 1))
chk("sep_flag", as.numeric(sp$separated), A$sep_flag)

ss <- morie_esl_sis_screening(Xs, ys, d = 4)
chk("sis_omega", ss$omega, A$sis_omega)
chk("sis_sel",   ss$selected, A$sis_sel)

wl <- morie_esl_wavelet_smooth(noisy)
chk("wlt",      c(wl$sigma, wl$threshold), A$wlt)
chk("wlt_sig5", wl$signal[1:5], A$wlt_sig5)

gm <- morie_esl_em_gmm(xm, k = 2, seed = 1L)
chk("gmm_mono", min(diff(gm$loglik_path)) > -1e-8, 1)
cat(sprintf("  note: gmm loglik R=%.6f python=%.6f (independent inits)\n",
            gm$loglik, A$gmm_ll))
dn <- morie_esl_gaussian_mixture(xm, k = 2, newdata = c(-4, 0, 4), seed = 1L)
chk("gmm_dens_ord", as.numeric(dn$density[1] > dn$density[2]), 1)

cv <- morie_esl_cv_score(Xcv, ycv, k = 5L)
chk("cv_ok", as.numeric(cv$cv < 0.2 * stats::var(ycv)), A$cv_ok)

cat(sprintf("\nESL parity: %d passed, %d failed\n", pass, fail))
if (fail > 0) quit(status = 1)
