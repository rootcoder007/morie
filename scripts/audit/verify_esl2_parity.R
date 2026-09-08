# SPDX-License-Identifier: AGPL-3.0-or-later
# Base-R parity harness: morie R ESL part-2 functions vs anchors from live Python.
args <- commandArgs(trailingOnly = TRUE)
D <- args[length(args)]
for (f in args[-length(args)]) source(f)

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
    cat(sprintf("  FAIL %-16s maxdiff=%s\n    got =%s\n    want=%s\n", label,
      if (length(got) == length(want)) format(max(abs(got - want)), digits = 4) else "LENGTH",
      paste(signif(utils::head(got, 6), 8), collapse = ","),
      paste(signif(utils::head(want, 6), 8), collapse = ",")))
  }
}

Xsv <- rd("Xsv"); ysv <- v("ysv"); Xlar <- rd("Xlar"); ylar <- v("ylar")
Xpca <- rd("Xpca"); P <- rd("P"); ztps <- v("ztps"); Xica <- rd("Xica")
Xiso <- rd("Xiso"); tt <- v("tt"); Xlle <- rd("Xlle")
Xnn <- rd("Xnn"); ynn <- v("ynn"); Xbp <- rd("Xbp"); ybp <- v("ybp")
Wa <- rd("Wa"); Wb <- rd("Wb"); Xpd <- rd("Xpd"); Vrbm <- rd("Vrbm"); Xsm <- rd("Xsm")

## --- SVM / SVC: SMO draws its working index at random, so the exact alpha
## --- vector is RNG-dependent across languages. The invariants are not.
sv <- morie_esl_svm_kernel(Xsv, ysv, C = 1, kernel = "linear", seed = 1L)
chk("svm_acc", sv$accuracy, A$svm_acc)
chk("svm_gap", sv$dual_gap_check, A$svm_gap, tol = 1e-9)
chk("svm_box", as.numeric(max(sv$alpha) <= 1 + 1e-9), 1)
vc <- morie_esl_svc(Xsv, ysv, C = 1, seed = 1L)
chk("svc_acc", vc$accuracy, A$svc_acc)
chk("svc_margin_pos", as.numeric(vc$margin > 0), A$svc_margin_pos)

lr <- morie_esl_least_angle_reg(Xlar, ylar)
chk("lar_active", lr$active[1:3], A$lar_active)
chk("lar_coef",   lr$coef, A$lar_coef)
chk("lar_r2",     lr$r_squared, A$lar_r2)
chk("lar_equi",   diff(range(abs(lr$correlations[3, lr$active[1:2]]))), A$lar_equi, tol = 1e-7)

sp <- morie_esl_sparse_pca(Xpca, k = 1, lambda_ = 0)
chk("spca_load_abs", abs(sp$loadings[, 1]), A$spca_load_abs, tol = 1e-6)
chk("spca_sparse2", as.numeric(morie_esl_sparse_pca(Xpca, k = 1, lambda_ = 2)$sparsity > 0),
    A$spca_sparse2)

tp <- morie_esl_thin_plate_spline(P, ztps, lambda_ = 0)
chk("tps_resid_max", max(abs(tp$residuals)), A$tps_resid_max, tol = 1e-6)
tpb <- morie_esl_thin_plate_spline(P, ztps, lambda_ = 1e8)
Am <- cbind(1, P); plane <- as.numeric(Am %*% qr.solve(Am, ztps))
chk("tps_plane_max", max(abs(tpb$fitted - plane)), A$tps_plane_max, tol = 1e-4)
chk("tps_edf_big",   tpb$edf, A$tps_edf_big, tol = 1e-6)

ic <- morie_esl_ica(Xica, k = 2, seed = 1L)
chk("ica_sd", apply(ic$sources, 2, function(z) sqrt(mean((z - mean(z))^2))), A$ica_sd, tol = 1e-6)
S <- cbind(sin(seq(0, 8 * pi, length.out = 600)), sign(cos(2.7 * seq(0, 8 * pi, length.out = 600))))
Cm <- abs(stats::cor(ic$sources, S))
chk("ica_match", min(apply(Cm, 1, max)) > 0.9, 1)

iso <- morie_esl_isomap(Xiso, k = 2, neighbors = 8)
chk("iso_corr", abs(stats::cor(iso$embedding[, 1], tt)), A$iso_corr, tol = 1e-6)
Deu <- sqrt(pmax(outer(rowSums(Xiso^2), rowSums(Xiso^2), "+") - 2 * tcrossprod(Xiso), 0))
# The squared-expansion distance formula loses ~1e-7 absolute on distances of
# order 30; that is float noise, not a shortcut through the manifold.
chk("iso_geo_ge", as.numeric(all(iso$geodesic >= Deu - 1e-6 * pmax(Deu, 1))),
    A$iso_geo_ge)

ll <- morie_esl_lle(Xlle, k = 2, neighbors = 10)
chk("lle_wsum", max(abs(rowSums(ll$weights) - 1)), A$lle_wsum, tol = 1e-8)

nn <- morie_esl_neural_net(Xnn, ynn, M = 8, lr = 0.3, n_epochs = 2000L, seed = 1L)
# Fixed-step gradient descent is only monotone when lr < 2/L, which depends on
# the initialisation -- so assert overall decrease, which always holds here.
chk("nn_decreases", as.numeric(nn$loss_path[length(nn$loss_path)] < nn$loss_path[1] / 10), 1)
chk("nn_fits",  as.numeric(nn$r_squared > 0.8), 1)

bp <- morie_esl_backprop(Xbp, ybp, list(alpha = Wa, alpha0 = numeric(4),
                                        beta = Wb, beta0 = 0))
# R flattens column-major, numpy .ravel() is row-major -- transpose to compare.
chk("bp_ga",   as.numeric(t(bp$grad_alpha)), A$bp_ga)
chk("bp_loss", bp$loss, A$bp_loss)

f <- function(Z) 3 * Z[, 1] + Z[, 2]^2
pdv <- morie_esl_partial_dependence(f, Xpd, S = 1, n_grid = 9L)
chk("pd_slope",  stats::coef(stats::lm(pdv$pd ~ pdv$grid[, 1]))[[2]], A$pd_slope, tol = 1e-6)
chk("pd_center", abs(mean(pdv$centered)), A$pd_center, tol = 1e-9)

rb <- morie_esl_boltzmann(Vrbm, h = 3, lr = 0.5, n_epochs = 200L, seed = 1L)
chk("rbm_drop", as.numeric(rb$error_path[length(rb$error_path)] < rb$error_path[1]),
    A$rbm_drop)

dp <- morie_esl_dirichlet_proc(alpha = 2, n_atoms = 200L, seed = 1L)
chk("dp_sum", abs(sum(dp$weights) + dp$truncation_mass - 1), A$dp_sum, tol = 1e-10)

mrf <- morie_esl_markov_rf(rbind(c(1, 2), c(2, 3)))
chk("mrf_marg", as.numeric(t(mrf$marginals)), A$mrf_marg)
chk("mrf_logZ", mrf$log_Z, A$mrf_logZ)
chk("mrf_mode", as.numeric(mrf$mode), c(0, 0, 0))

J <- sapply(c(0, 1, 2, 3), function(m)
  morie_esl_score_match(function(Z) -(Z - m), Xsm)$objective)
chk("sm_argmin", which.min(J), A$sm_argmin)
chk("sm_trace",  morie_esl_score_match(function(Z) -(Z - 2), Xsm)$trace_term,
    A$sm_trace, tol = 1e-4)

cat(sprintf("\nESL part-2 parity: %d passed, %d failed\n", pass, fail))
if (fail > 0) quit(status = 1)
