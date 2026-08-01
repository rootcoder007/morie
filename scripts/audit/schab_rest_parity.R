# Three-way parity: the R arm against values computed by the Python arm,
# for the last five Schabenberger shelf modules.
source("/home/rootcoder/work/morie/r-package/morie/R/aaa_schab_rest_shared.R")

py <- jsonlite::fromJSON("/tmp/py_rest_values.json")
FAIL <- character(0)

chk <- function(name, got, want, tol = 1e-12) {
  got <- as.numeric(got); want <- as.numeric(want)
  if (length(got) != length(want)) {
    cat(sprintf("  %-54s FAIL (length %d vs %d)\n", name, length(got),
                length(want)))
    FAIL <<- c(FAIL, name); return(invisible())
  }
  d <- max(abs(got - want))
  rel <- max(abs(got - want) / pmax(abs(want), 1e-300))
  ok <- d <= tol || rel <= tol
  cat(sprintf("  %-54s %s  max|d|=%.3e\n", name, if (ok) "PASS" else "FAIL", d))
  if (!ok) FAIL <<- c(FAIL, name)
}

rook <- function(g) {
  n <- g * g
  w <- matrix(0, n, n)
  for (i in 0:(g - 1)) for (j in 0:(g - 1)) {
    k <- i * g + j + 1
    for (dd in list(c(0, 1), c(1, 0), c(0, -1), c(-1, 0))) {
      a <- i + dd[1]; b <- j + dd[2]
      if (a >= 0 && a < g && b >= 0 && b < g) w[k, a * g + b + 1] <- 1
    }
  }
  w
}

cat("\n[Ch 1: Moran moments on the Example 1.7 design]\n")
W <- rook(10)
m <- .schab_moran_moments(py$moran_z, W)
for (k in c("I", "expectation", "variance_normal", "variance_randomization",
            "sd_normal", "sd_randomization", "z_normal", "z_randomization",
            "kurtosis_b", "S0", "S1", "S2", "geary_c")) {
  chk(paste0("moran$", k), m[[k]], py$moran[[k]], 1e-10)
}

cat("\n[Ch 3: cross-K, eq (3.9)]\n")
p1 <- matrix(py$pp$p1, ncol = 2)
p2 <- matrix(py$pp$p2, ncol = 2)
reg <- c(0, 0, 1, 1)
r <- py$pp$r
wp <- matrix(py$pp$wp, ncol = 2)
chk("vectorised Ripley weights", .schab_ripley_weights(wp, reg, py$pp$wt),
    py$pp$weights, 1e-12)
ck <- .schab_cross_k_combined(p1, p2, reg, r)
chk("K*", ck$K_star, py$pp$K_star, 1e-10)
chk("Khat_12", ck$K_12, py$pp$K_12, 1e-10)
chk("Khat_21", ck$K_21, py$pp$K_21, 1e-10)
chk("L* - h", ck$L_minus_h, py$pp$L_minus_h, 1e-10)
chk("uncorrected Khat_12",
    .schab_cross_k(p1, p2, reg, r, "none"), py$pp$K12_uncorrected, 1e-10)
dd <- .schab_dc_d(p1, p2, reg, r)
chk("Diggle-Chetwynd D(h)", dd$D, py$pp$D, 1e-10)
chk("K_11 (border)", dd$K_11, py$pp$K_11, 1e-10)
chk("K_22 (border)", dd$K_22, py$pp$K_22, 1e-10)

cat("\n[Ch 4: periodogram, eqs (4.57)-(4.59)]\n")
zz <- matrix(py$spec$z, nrow = py$spec$r)
P <- .schab_periodogram(zz)
Q <- .schab_periodogram_from_cov(zz)
sc <- .schab_sample_cov2d(zz)
chk("Fourier frequencies omega1", P$omega1, py$spec$omega1)
chk("Fourier frequencies omega2", P$omega2, py$spec$omega2)
chk("periodogram (mean removed)", as.numeric(P$periodogram),
    py$spec$periodogram, 1e-12)
chk("Fourier transform of Chat", as.numeric(Q$periodogram),
    py$spec$from_cov, 1e-12)
chk("sample covariance Chat", as.numeric(sc$cov), py$spec$cov, 1e-12)
if (!identical(P$mean_invariant, py$spec$mean_invariant)) {
  cat("  mean-invariance flag                                   FAIL\n")
  FAIL <- c(FAIL, "mean invariance flag")
} else {
  cat("  mean-invariance flag                                   PASS\n")
}

cat("\n[Ch 8: point source + moving windows]\n")
s <- matrix(py$ns$s, ncol = 2)
psc <- .schab_point_source_corr(s, py$ns$src, 0.35, 0.12, 0.07)
chk("eq (8.1) correlation matrix", as.numeric(psc$correlation),
    py$ns$corr, 1e-12)
chk("source distances c_i", psc$source_distance, py$ns$ci, 1e-12)
chk("minimum eigenvalue", psc$min_eigenvalue, py$ns$min_eig, 1e-9)
chk("practical range 3/theta1", .schab_practical_range(0.35), py$ns$pr)
chk("pairwise practical range",
    .schab_practical_range(0.35, 0.12, 0.07, ci = 2.0, cj = 3.5),
    py$ns$pr_pair)

big <- matrix(py$ns$big, ncol = 2)
tg <- matrix(py$ns$tg, ncol = 2)
hw <- .schab_haas_window(big, tg[1, ], min_sites = 35L, step = 5L)
chk("Haas window size", hw$n_sites, py$ns$haas_n, 0)
chk("Haas window radius", hw$radius, py$ns$haas_radius, 1e-12)
chk("Haas lag counts", hw$lag_counts, py$ns$haas_counts, 0)
chk("Haas member set", sort(hw$index - 1L), py$ns$haas_index_sorted, 0)

mw <- .schab_moving_window_krige(big, py$ns$zbig, tg, min_sites = 35L,
                                 step = 5L, local_variogram = TRUE)
lk <- .schab_moving_window_krige(big, py$ns$zbig, tg, min_sites = 35L,
                                 step = 5L, local_variogram = FALSE)
chk("moving-window predictions", mw$prediction, py$ns$mw_pred, 1e-8)
chk("per-window sills", mw$local_sill, py$ns$mw_sill, 1e-8)
chk("per-window ranges", mw$local_range, py$ns$mw_range, 1e-8)
chk("window sizes", mw$window_sizes, py$ns$mw_sizes, 0)
chk("local-kriging predictions (global theta)", lk$prediction,
    py$ns$lk_pred, 1e-8)
chk("local kriging keeps the global range", lk$local_range,
    py$ns$lk_range, 1e-8)
chk("global sill", mw$global_sill, py$ns$global_sill, 1e-8)
chk("global range", mw$global_range, py$ns$global_range, 1e-8)

cat(sprintf("\n%s  %d failed\n", strrep("=", 62), length(FAIL)))
if (length(FAIL)) {
  cat("FAILED:", paste(FAIL, collapse = ", "), "\n")
  quit(status = 1)
}
