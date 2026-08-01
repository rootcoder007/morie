"""Emit Python-arm values for the Morin R parity check."""
import json
import math

import numpy as np

from morie.fn import _morin

out = {}
out["partial_perm_10_3"] = _morin.partial_permutations(10, 3)
out["multinomial_3_2_5"] = _morin.multinomial_coefficient([3, 2, 5])
out["multinomial_3_2_5_of_16"] = _morin.multinomial_coefficient([3, 2, 5], 16)
out["stars_bars_2_3"] = _morin.stars_and_bars(2, 3)
hs = _morin.hockey_stick(10, 4)
out["hockey_10_4"] = list(hs)
out["or_general_king_heart"] = _morin.prob_or_general(1/13, 1/4, 1/52)
post, pz = _morin.bayes_general([0.02, 0.98], [0.95, 0.10])
out["bayes_posteriors"] = [float(x) for x in post]
out["bayes_pz"] = pz
v, mu = _morin.pmf_variance([1, 2, 3, 4, 5, 6], [1/6] * 6)
out["die_var_mean"] = [v, mu]
cv, cp = _morin.pmf_sum_convolution([1, 2], [0.5, 0.5], [1, 2, 3], [1/3] * 3)
out["conv_values"] = [float(x) for x in cv]
out["conv_probs"] = [float(x) for x in cp]
x8 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
out["popvar_8"] = _morin.population_variance(x8)
out["samplevar_8"] = _morin.sample_variance(x8)
out["sd_sum_3_4"] = _morin.sd_sum_independent([3.0, 4.0])
out["sd_mean_hetero_3_4"] = _morin.sd_of_mean_hetero([3.0, 4.0])
out["binom_pmf_2_4_half"] = _morin.binomial_pmf(2, 4, 0.5)
out["binom_pmf_big"] = _morin.binomial_pmf(20000, 100000, 0.2)
out["binom_second_20_quarter"] = _morin.binomial_second_moment(20, 0.25)
out["p01_9"] = _morin.p_zero_equals_one(9)
out["pois_pmf_3_2"] = _morin.poisson_pmf(3, 2.0)
pm, pv = _morin.poisson_mean_var(4.2)
out["pois_meanvar_42"] = [pm, pv]
out["pois_mode_7"] = _morin.poisson_mode(7.0)
out["hyper_2_52_13_5"] = _morin.hypergeometric_pmf(2, 52, 13, 5)
out["exp_density_05_2"] = _morin.exponential_waiting_density(0.5, 2.0)
out["exp_interval_1_001_2"] = _morin.exponential_interval_probability(1.0, 0.01, 2.0)
out["exp_crossing"] = _morin.exponential_crossing_time(0.2, 0.05, 4.0)
out["gauss_2n_0_50"] = _morin.gaussian_approx_2n(0.0, 50)
out["gauss_n_2_100"] = _morin.gaussian_approx_n(2.0, 100)
out["gauss_biased_0_10000_03"] = _morin.gaussian_approx_biased(0.0, 10000, 0.3)
out["pois_gauss_420_400"] = _morin.poisson_gaussian(420, 400.0)
sd, mu531 = _morin.pmf_sd([2.0, 3.2, 7.0], [0.6, 0.1, 0.3])
out["sd531"] = [sd, mu531]
muy, sy, r = _morin.linear_model_stats(1.0, 0.0, 7.5, 0.0, 10.6)
out["model_676"] = [muy, sy, r]
out["excess_05"] = _morin.excess_score_factor(0.5)
X5 = [2.0, 3.0, 3.0, 5.0, 7.0]
Y5 = [1.0, 1.0, 3.0, 4.0, 6.0]
A, B, S = _morin.least_squares_fit(X5, Y5)
out["ls_ABS"] = [A, B, S]
out["ls_r"] = _morin.sample_r(X5, Y5)
Ar, Cr, rr = _morin.regression_slope_product(X5, Y5)
out["slope_product"] = Ar * Cr
g = np.linspace(-16.0, 16.0, 3201)
dx = np.exp(-g ** 2 / 2) / math.sqrt(2 * math.pi)
dy = np.exp(-g ** 2 / 8) / math.sqrt(2 * math.pi * 4)
out["sumdens_1"] = _morin.sum_density_convolution(g, dx, g, dy, 1.0)
out["gauss_sum_1"] = _morin.gaussian_sum_density(1.0, 1.0, 2.0)
e1, a1, v1 = _morin.one_plus_a_to_n(-1 / 365, 23, order=1)
out["ladder1"] = [e1, a1, v1]
e2, a2, v2 = _morin.one_plus_a_to_n(0.05, 200, order=2)
out["ladder2"] = [e2, a2, v2]
q, d = _morin.power_derivative_quotient(2.0, 5, 1e-7)
out["quotient"] = [q, d]

print(json.dumps(out))
