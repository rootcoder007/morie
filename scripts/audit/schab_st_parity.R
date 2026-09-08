# Three-way parity: R arm vs values computed by the Python arm.
# Expected values are read from /tmp/py_st_values.json, produced by the
# Python side on the same inputs.
source("/home/rootcoder/work/morie/r-package/morie/R/aaa_schab_st_shared.R")

py <- jsonlite::fromJSON("/tmp/py_st_values.json")
FAIL <- character(0)

chk <- function(name, got, want, tol = 1e-12) {
  got <- as.numeric(got); want <- as.numeric(want)
  if (length(got) != length(want)) {
    cat(sprintf("  %-52s FAIL (length %d vs %d)\n", name, length(got), length(want)))
    FAIL <<- c(FAIL, name); return(invisible())
  }
  d <- max(abs(got - want))
  rel <- max(abs(got - want) / pmax(abs(want), 1e-300))
  ok <- d <= tol || rel <= tol
  cat(sprintf("  %-52s %s  max|d|=%.3e\n", name, if (ok) "PASS" else "FAIL", d))
  if (!ok) FAIL <<- c(FAIL, name)
}

cs <- function(h) 2 * exp(-h / 3)
ct <- function(k) 1.5 * exp(-k / 2)
h <- py$h; k <- py$k

cat("\n[9.2] separable\n")
chk("product", .schab_st_separable_covariance(h, k, cs, ct, "product"), py$product)
chk("sum", .schab_st_separable_covariance(h, k, cs, ct, "sum"), py$sum)
chk("product_sum", .schab_st_separable_covariance(h, k, cs, ct, "product_sum"),
    py$product_sum)
chk("eq (9.4) exponential separable",
    .schab_st_exponential_separable(h, k, 0.7, 0.3), py$exp_sep)

cat("\n[9.3] non-separable\n")
chk("Gneiting eq (9.8)",
    .schab_st_gneiting(h, k, sigma2 = 2, a = 0.5, c = 0.3, alpha = 1,
                       beta = 0.8, gamma = 1, d = 2), py$gneiting)
chk("Gneiting eq (9.9)",
    .schab_st_gneiting_with_temporal(h, k, sigma2 = 2, a = 0.5, c = 0.3,
                                     alpha = 1, beta = 0.5, beta_t = 0.4,
                                     gamma = 1, d = 2), py$gneiting_t)
chk("power mixture, Poisson",
    .schab_st_power_mixture(py$rs, py$rt, "poisson", lam = 2), py$pm_poisson)
chk("power mixture, binomial",
    .schab_st_power_mixture(py$rs, py$rt, "binomial", n = 4, pi = 0.3),
    py$pm_binom)
chk("bivariate power mixture eq (9.13)",
    .schab_st_bivariate_power_mixture(py$rs, py$rt, py$pmf), py$pm_biv)
chk("scale mixture eq (9.16)",
    .schab_st_scale_mixture(h, k, cs, ct, py$nodes, py$weights), py$scale_mix)

cat("\n[9.3.4] quadrature and Bessel\n")
gl <- .schab_gauss_legendre(24)
chk("Gauss-Legendre nodes", gl$nodes, py$gl_nodes)
chk("Gauss-Legendre weights", gl$weights, py$gl_weights)
chk("J0", .schab_bessel_j0(py$bessel_x), py$j0)
chk("K1", .schab_bessel_k1(py$bessel_z), py$k1)
chk("Whittle covariance",
    .schab_whittle_covariance(py$whittle_h, sigma2 = 3, theta = 1), py$whittle)
jz <- .schab_st_jones_zhang(py$jz_h, py$jz_k, sigma2 = 1, theta = 1, c = 1,
                            p = 2, d = 2, n_quad = 40)
chk("Jones-Zhang eq (9.17)", jz$covariance, py$jz, tol = 1e-10)

cat("\n[9.4] semivariogram\n")
covfn <- function(d, u) .schab_st_separable_covariance(d, u, cs, ct)
chk("gamma = C(0,0) - C(h,k)",
    .schab_st_semivariogram_from_cov(h, k, covfn), py$gamma_model)

coords <- matrix(py$coords, ncol = 2)
times <- py$times
z <- py$z
emp <- .schab_st_empirical_semivariogram(coords, times, z, 4L, 3L)
chk("empirical (9.18) counts", as.numeric(emp$counts), as.numeric(py$emp_counts), 0)
g <- as.numeric(emp$gamma); gw <- as.numeric(py$emp_gamma)
both <- !is.na(g) & !is.na(gw)
chk("empirical (9.18) gamma", g[both], gw[both])
cond <- .schab_st_conditional_semivariogram(coords, times, z, py$at_time, 3L)
chk("conditional (9.19)", cond$gamma[!is.na(cond$gamma)],
    py$cond_gamma[!is.na(py$cond_gamma)])
chk("WLS objective",
    .schab_st_wls_objective(emp, function(hh, kk)
      .schab_st_semivariogram_from_cov(hh, kk, covfn)), py$wls)

cat("\n[9.5] point processes\n")
pts <- matrix(py$pts, ncol = 2)
tt <- py$tt
lam <- .schab_st_intensity(pts, tt, py$region, py$span)
chk("intensity eq (9.20)", lam$intensity, py$intensity)
mg <- .schab_st_marginal_intensities(pts, tt, py$region, py$span, 4L, 4L)
chk("marginal spatial", as.numeric(mg$marginal_spatial), as.numeric(py$marg_s))
chk("marginal temporal", as.numeric(mg$marginal_temporal), as.numeric(py$marg_t))
cst <- .schab_cstr_test(pts, tt, py$region, py$span, 3L, 3L)
chk("CSTR index of dispersion", cst$index_of_dispersion, py$cstr_index)
chk("CSTR p-value", cst$p_value, py$cstr_p)
chk("chi2 survival", vapply(py$chi_x, function(x)
  .schab_st_chi2_sf(x, 1), numeric(1)), py$chi_sf)
sp <- .schab_st_separability_test(100, 103)
chk("separability p (halved)", sp$p_value, py$sep_p)

cat(sprintf("\n%s  %d failed\n", strrep("=", 62), length(FAIL)))
if (length(FAIL)) {
  cat("FAILED:", paste(FAIL, collapse = ", "), "\n")
  quit(status = 1)
}
