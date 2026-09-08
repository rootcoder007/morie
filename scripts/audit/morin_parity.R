# Morin shelf parity: R mirror vs Python-arm values.
args <- commandArgs(trailingOnly = TRUE)
py <- jsonlite::fromJSON(args[[1]])
src <- args[[2]]
source(file.path(src, "morin_native.R"))

fails <- 0L
cmp <- function(name, r_val, py_val, tol = 1e-12) {
  r_val <- as.numeric(unlist(r_val))
  py_val <- as.numeric(unlist(py_val))
  if (length(r_val) != length(py_val) ||
        any(abs(r_val - py_val) > tol * pmax(1, abs(py_val)))) {
    cat(sprintf("FAIL %s\n  R : %s\n  Py: %s\n", name,
                paste(format(r_val, digits = 17), collapse = ", "),
                paste(format(py_val, digits = 17), collapse = ", ")))
    fails <<- fails + 1L
  } else {
    cat(sprintf("ok   %s\n", name))
  }
}

cmp("partial_perm", morie_partial_permutations(10, 3), py$partial_perm_10_3)
cmp("multinomial", morie_multinomial_coef(c(3, 2, 5)), py$multinomial_3_2_5)
cmp("multinomial16", morie_multinomial_coef(c(3, 2, 5), 16), py$multinomial_3_2_5_of_16)
cmp("stars_bars", morie_stars_bars(2, 3), py$stars_bars_2_3)
hs <- morie_hockey_stick(10, 4)
cmp("hockey", c(hs$sum, hs$binomial), py$hockey_10_4)
cmp("or_general", morie_prob_rules(1/13, 1/4, 1/52)$or_general, py$or_general_king_heart)
b <- morie_bayes(c(0.02, 0.98), c(0.95, 0.10))
cmp("bayes_post", b$posteriors, py$bayes_posteriors)
cmp("bayes_pz", b$p_z, py$bayes_pz)
m <- morie_pmf_moments(1:6, rep(1/6, 6))
cmp("die_var_mean", c(m$variance, m$mean), py$die_var_mean)
cv <- morie_pmf_convolve(1:2, c(0.5, 0.5), 1:3, rep(1/3, 3))
cmp("conv_values", cv$values, py$conv_values)
cmp("conv_probs", cv$probs, py$conv_probs)
x8 <- c(2, 4, 4, 4, 5, 5, 7, 9)
dv <- morie_data_variance(x8)
cmp("popvar", dv$population_variance, py$popvar_8)
cmp("samplevar", dv$sample_variance, py$samplevar_8)
sf <- morie_sd_forms(c(3, 4))
cmp("sd_sum", sf$sd_sum, py$sd_sum_3_4)
cmp("sd_mean_hetero", sf$sd_mean_hetero, py$sd_mean_hetero_3_4)
cmp("binom_small", morie_binomial_dist(4, 0.5, k = 2)$pmf, py$binom_pmf_2_4_half)
# lgamma(100001) differs between R's and libm's implementations at the
# ~1e-10 relative level; both arms are log-space-correct, so tolerate it.
cmp("binom_big", morie_binomial_dist(100000, 0.2, k = 20000)$pmf, py$binom_pmf_big,
    tol = 1e-8)
cmp("binom_second", morie_binomial_dist(20, 0.25)$second_moment, py$binom_second_20_quarter)
cmp("p01", morie_binomial_dist(9, 0.5)$p_zero_equals_one, py$p01_9)
cmp("pois_pmf", morie_poisson_dist(2.0, k = 3)$pmf, py$pois_pmf_3_2)
pd <- morie_poisson_dist(4.2)
cmp("pois_meanvar", c(pd$mean, pd$variance), py$pois_meanvar_42, tol = 1e-9)
cmp("pois_mode", morie_poisson_dist(7.0)$mode, py$pois_mode_7)
cmp("hyper", morie_hypergeometric_dist(2, 52, 13, 5)$pmf, py$hyper_2_52_13_5)
ed <- morie_exponential_dist(2.0, t = 0.5)
cmp("exp_density", ed$density, py$exp_density_05_2)
cmp("exp_interval", morie_exponential_dist(2.0, t = 1.0, dt = 0.01)$interval_probability,
    py$exp_interval_1_001_2)
cmp("exp_crossing", morie_exponential_dist(0.2, rate_slow = 0.05, ratio = 4.0)$crossing_time,
    py$exp_crossing)
cmp("gauss_2n", morie_gaussian_approx(0, n = 50, form = "two_n"), py$gauss_2n_0_50)
cmp("gauss_n", morie_gaussian_approx(2, n = 100, form = "n"), py$gauss_n_2_100)
cmp("gauss_biased", morie_gaussian_approx(0, n = 10000, p = 0.3, form = "biased"),
    py$gauss_biased_0_10000_03)
cmp("pois_gauss", morie_gaussian_approx(420 - 400, a = 400, form = "poisson"),
    py$pois_gauss_420_400)
m531 <- morie_pmf_moments(c(2, 3.2, 7), c(0.6, 0.1, 0.3))
cmp("sd531", c(m531$sd, m531$mean), py$sd531)
lm676 <- morie_linear_corr_model(1, 7.5, 10.6)
cmp("model_676", c(lm676$mu_y, lm676$sigma_y, lm676$r), py$model_676)
# m = 1, sigma_x = 1, sigma_z = sqrt(3) gives r = 1/2 exactly
cmp("excess", morie_linear_corr_model(1, 1, sqrt(3))$excess_factor, py$excess_05)
X5 <- c(2, 3, 3, 5, 7); Y5 <- c(1, 1, 3, 4, 6)
ls5 <- morie_least_squares(X5, Y5)
cmp("ls_ABS", c(ls5$A, ls5$B, ls5$S), py$ls_ABS)
cmp("ls_r", ls5$r, py$ls_r)
cmp("slope_product", ls5$slope_product, py$slope_product)
g <- seq(-16, 16, length.out = 3201)
dx <- exp(-g^2 / 2) / sqrt(2 * pi)
dy <- exp(-g^2 / 8) / sqrt(2 * pi * 4)
sd1 <- morie_sum_density(g, dx, g, dy, 1.0, sigma_x = 1, sigma_y = 2)
cmp("sumdens", sd1$density, py$sumdens_1, tol = 1e-9)
cmp("gauss_sum", sd1$gaussian_closed_form, py$gauss_sum_1)
l1 <- morie_approx_ladder(-1/365, 23, order = 1)
cmp("ladder1", c(l1$exact, l1$approx, l1$validity), py$ladder1)
l2 <- morie_approx_ladder(0.05, 200, order = 2)
cmp("ladder2", c(l2$exact, l2$approx, l2$validity), py$ladder2)
lq <- morie_approx_ladder(0, 1, x = 2, delta = 1e-7, power = 5)
cmp("quotient", c(lq$quotient, lq$derivative), py$quotient, tol = 1e-9)

cat(sprintf("\n== %d failed\n", fails))
if (fails > 0L) quit(status = 1L)
