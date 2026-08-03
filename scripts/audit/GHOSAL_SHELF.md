# Ghosal & van der Vaart (2017) shelf — 260 modules
#
# Fundamentals of Nonparametric Bayesian Inference, CUP.
# PDF: data/datasets/userguides/other/pdf/ (library; PDF is truth).
# Status: tranche-by-tranche implementation, three-way parity.

| section | module | function | formula (stub claim, to verify vs PDF) | keys |
|---|---|---|---|---|
| Ghosal & van der Vaart (2017), Ch 1, E | ghs001 | ghosal_ch1_bayes_formula | Pi(B \| X) = ( int_B p_theta(X) dPi(theta) ) / ( int p_theta(X) dPi(theta) ) | posterior |
| Ghosal & van der Vaart (2017), Ch 2, E | ghs002 | ghosal_ch2_random_basis_expansion | f = sum_{j in J} beta_j * psi_j | function |
| Ghosal & van der Vaart (2017), Ch 2, E | ghs003 | ghosal_ch2_basis_truncation_error | \|\| f - sum_{j=1}^{J} f_j * psi_j \|\| <~ (1/J)^(alpha/k) * \|\|f\|\|_alpha^* | value |
| Ghosal & van der Vaart (2017), Ch 2, E | ghs004 | ghosal_ch2_exponential_link_density | p_f(x) = exp( f(x) - c(f) ),   c(f) = log integral e^f d mu | distribution |
| Ghosal & van der Vaart (2017), Ch 2, E | ghs005 | ghosal_ch2_location_scale_mixture_limit | integral (1/sigma) * psi( ( . - mu ) / sigma ) f(mu) d mu  ->  f(.),   as sigma -> 0 | distribution |
| Ghosal & van der Vaart (2017), Ch 2, E | ghs006 | ghosal_ch2_feller_density_approximation | a(x; k, F) = integral h_k(x; z) dF(z),   h_k(x; z) = (k / V(x)) * integral_{[z, infty)} (t - x) g_k( | value |
| Ghosal & van der Vaart (2017), Ch 2, E | ghs007 | ghosal_ch2_binary_regression_density | p_f(y \| x) = H(f(x))^y * (1 - H(f(x)))^(1 - y),   y in {0, 1} | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs008 | ghosal_ch3_normalized_weights_prior | p_k = Y_k / sum_{j=1}^{infty} Y_j,   k in N | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs009 | ghosal_ch3_stick_breaking_weights | p_j = ( prod_{l=1}^{j-1} (1 - V_l) ) * V_j | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs010 | ghosal_ch3_discrete_hazard_rate | V_j = p_j / (1 - sum_{l=1}^{j-1} p_l) = P(X = j \| X >= j) | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs011 | ghosal_ch3_countable_dirichlet_marginal | (p_1, ..., p_k, 1 - sum_{j=1}^{k} p_j) ~ Dir(k+1; alpha_1, ..., alpha_k, sum_{j=k+1}^{infty} alpha_j | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs012 | ghosal_ch3_countable_dirichlet_posterior_l | Dir( l+1; alpha_1 + N_1, ..., alpha_l + N_l, sum_{j=l+1}^{infty} alpha_j + n - sum_{j=1}^{l} N_j ) | posterior |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs013 | ghosal_ch3_countable_dirichlet_posterior_k | Dir( k+1; alpha_1 + N_1, ..., alpha_k + N_k, sum_{j=k+1}^{infty} alpha_j + n - sum_{j=1}^{k} N_j ) | posterior |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs014 | ghosal_ch3_dirichlet_posterior_mean | E(p_j \| X_1, ..., X_n) = (alpha_j + N_j) / ( sum_{l=1}^{infty} alpha_l + n ) | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs015 | ghosal_ch3_dirichlet_posterior_var | var(p_j \| X_1, ..., X_n) = (alpha_j + N_j) * ( sum_{l != j} alpha_l + n - N_j ) / ( ( sum_{l=1}^{inf | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs016 | ghosal_ch3_dirichlet_posterior_cov | cov(p_j, p_{j'} \| X_1, ..., X_n) = - (alpha_j + N_j) * (alpha_{j'} + N_{j'}) / ( ( sum_{l=1}^{infty} | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs017 | ghosal_ch3_discrete_random_measure | P = sum_{i=1}^{infty} W_i * delta_{theta_i} | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs018 | ghosal_ch3_tree_splitting_variables | V_{epsilon 0} = P(A_{epsilon 0} \| A_epsilon),   V_{epsilon 1} = P(A_{epsilon 1} \| A_epsilon) | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs019 | ghosal_ch3_tree_set_probability | P(A_{epsilon_1 ... epsilon_m}) = V_{epsilon_1} * V_{epsilon_1 epsilon_2} * ... * V_{epsilon_1 ... ep | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs020 | ghosal_ch3_tree_countable_additivity | E[ V_epsilon * V_{epsilon 0} * V_{epsilon 0 0} * ... ] = 0   for all epsilon in E*,   and E[ V_1 * V | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs021 | ghosal_ch3_tailfree_max_bound | E[ max_{epsilon in E^m} P(A_epsilon) ]^2 <= sum_{epsilon in E^m} prod_{j=1}^{m} E(V_{epsilon_1 ... e | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs023 | ghosal_ch3_tailfree_abs_continuity_cond | sup_{m in N} max_{epsilon in E^m} E( prod_{j=1}^{m} V_{epsilon_1 ... epsilon_j}^2 ) / mu^2( A_{epsil | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs024 | ghosal_ch3_tailfree_canonical_summability | sum_{m=1}^{infty} max_{epsilon in E^m} \| E(V_epsilon) - 1/2 \| < infty,   sum_{m=1}^{infty} max_{epsi | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs025 | ghosal_ch3_tailfree_density_product | p(x) = prod_{j=1}^{infty} ( 2 * V_{x_1 x_2 ... x_j} ) | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs026 | ghosal_ch3_tailfree_finite_density_pm | p_m = sum_{epsilon in E^m} ( P(A_epsilon) / mu(A_epsilon) ) * 1_{A_epsilon} | distribution |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs027 | ghosal_ch3_tailfree_strong_support_event | Pi( integral \| p / p_m - 1 \| d mu < epsilon / ( 2 \|\|p_0\|\|_infty + epsilon ) ) * Pi( \|\| p_m - p_0 \|\|_ | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs028 | ghosal_ch3_polya_tree_first_two_moments | E( P(A_{epsilon_1 ... epsilon_m}) ) = prod_{j=1}^{m} alpha_{epsilon_1 ... epsilon_j} / ( alpha_{epsi | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs029 | ghosal_ch3_polya_tree_density_moments | E( p(x) ) = prod_{j=1}^{infty} 2 * alpha_{x_1 ... x_j} / ( alpha_{x_1 ... x_{j-1} 0} + alpha_{x_1 .. | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs030 | ghosal_ch3_polya_tree_posterior_density | E( p(x) \| X_1, ..., X_n ) = prod_{j=1}^{infty} ( 2 * a_j + 2 * N_{x_1 ... x_j} ) / ( 2 * a_j + N_{x_ | posterior |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs031 | ghosal_ch3_polya_tree_mixture_post_density | E( p(x) \| theta, X_1, ..., X_n ) = g_theta(x) * prod_{j=1}^{infty} ( 2 * a_j + 2 * N_{G_theta(x)_1 . | posterior |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs032 | ghosal_ch3_polya_tree_density_bounds | prod_{j>m} ( 1 - n / (2 * a_j) )  <=  prod_{j>m} ( 2 * a_j + 2 * N_{G_theta(x)_1 ... G_theta(x)_j} ) | value |
| Ghosal & van der Vaart (2017), Ch 3, E | ghs033 | ghosal_ch3_polya_tree_mixture_second_kind | g_theta(x) = prod_{j=1}^{infty} 2 * alpha_{x_1 ... x_j}(theta) / ( alpha_{x_1 ... x_{j-1} 0}(theta)  | distribution |
| Ghosal App A | gh_ap_a1 | ghosal_weak_conv_def | P_n ->_w P iff E_{P_n}[f] -> E_P[f] for all bounded continuous f | estimate |
| Ghosal App A | gh_ap_a2 | ghosal_prohorov_metric | d_P(P,Q) = inf{eps>0: P(A) <= Q(A^eps)+eps for all closed A} | estimate |
| Ghosal App A | gh_ap_a3 | ghosal_tv_distance | d_TV(P,Q) = sup_{A measurable} \|P(A)-Q(A)\| = (1/2)\|\|p-q\|\|_1 | estimate |
| Ghosal App A | gh_ap_a4 | ghosal_hellinger_dist | d_H^2(P,Q) = 1 - integral sqrt(p*q) = 1 - rho_{1/2}(P,Q) | estimate |
| Ghosal App B | gh_ap_b1 | ghosal_kl_props | KL(P,Q)>=0, KL(P,Q)=0 iff P=Q, d_TV^2(P,Q) <= KL(P,Q)/2 (Pinsker) | estimate |
| Ghosal App B | gh_ap_b2 | ghosal_kl_variation | V_k(P,Q) = integral \|log(p/q)\|^k dP, V_1 = KL(P,Q) | estimate |
| Ghosal App B | gh_ap_b3 | ghosal_renyi_div | D_alpha(P\|\|Q) = (1/(alpha-1)) log integral p^alpha q^{1-alpha}, alpha in (0,1) | estimate |
| Ghosal App C | gh_ap_c1 | ghosal_covering_num | N(eps,T,d) = min{card(C): T subset union_{c in C} B(c,eps)} | estimate |
| Ghosal App C | gh_ap_c2 | ghosal_packing_num | D(eps,T,d) = max{card(S): d(s_i,s_j)>eps for all i!=j, S subset T} | estimate |
| Ghosal App C | gh_ap_c3 | ghosal_bracket_num | N[](eps,T,d) = min{#brackets: each f in T in some [l_i,u_i], d(l_i,u_i)<=eps} | estimate |
| Ghosal App E | gh_ap_e1 | ghosal_bernstein_poly | \|\|B_K(f) - f\|\|_infty <= C*omega(f, 1/sqrt(K)) for modulus of continuity omega | estimate |
| Ghosal App E | gh_ap_e2 | ghosal_spline_space | dim(S_{K,r}) = K + r + 1, approximation error \|\|f - s*\|\|_2 ~ K^{-s} for s-smooth f | estimate |
| Ghosal App E | gh_ap_e3 | ghosal_wavelet_mra | f = sum_k a_k phi_{j0,k} + sum_{j>=j0} sum_k d_{jk} psi_{jk} | estimate |
| Ghosal App F | gh_ap_f1 | ghosal_donsker_class | F is Donsker iff J[](1, F, L2) < infty (Dudley bracketing integral) | estimate |
| Ghosal App F | gh_ap_f2 | ghosal_glivenko | \|\|F_n - F\|\|_infty -> 0 a.s. for any F | estimate |
| Ghosal App G | gh_ap_g1 | ghosal_fin_dir_def | p(x_1..x_k) propto prod x_j^{alpha_j-1}, (x_j)>=0, sum x_j=1 | estimate |
| Ghosal App G | gh_ap_g2 | ghosal_dir_moments | E[X_j] = alpha_j/alpha_0, Cov[X_i,X_j] = -alpha_i*alpha_j/(alpha_0^2*(alpha_0+1)) for i!=j | estimate |
| Ghosal App G | gh_ap_g3 | ghosal_dir_marginal | X_{j1}+..+X_{jm} ~ Be(alpha_{j1}+..+alpha_{jm}, alpha_0 - sum alpha_{jl}) | estimate |
| Ghosal App H | gh_ap_h1 | ghosal_inv_gauss | f(x; alpha, gamma) = sqrt(gamma/(2*pi*x^3)) exp(-(gamma(x-alpha)^2)/(2*alpha^2*x)) | estimate |
| Ghosal App I | gh_ap_i1 | ghosal_gp_sample_cont | E\|f(x)-f(y)\|^p <= C*\|\|x-y\|\|^{1+alpha} => paths Holder-(alpha/p) continuous | estimate |
| Ghosal App I | gh_ap_i2 | ghosal_dudley_entropy | E[sup_f \|G_n(f)\|] <= C * J[](sigma_F, F) for empirical process G_n | estimate |
| Ghosal App I | gh_ap_i3 | ghosal_borell_tis | P(sup_t f(t) - E sup_t f(t) > u) <= exp(-u^2 / (2 sigma_f^2)) | estimate |
| Ghosal App J | gh_ap_j1 | ghosal_levy_ito | M = M_fixed + M_atom + M_diffuse, Laplace functional E[exp(-lambda*M(A))] | estimate |
| Ghosal App J | gh_ap_j2 | ghosal_crm_laplace | log E[exp(-integral f dM)] = -integral_0^infty integral_X (1-e^{-f(x)u}) nu(du,dx) | estimate |
| Ghosal App J | gh_crm_def | ghosal_completely_random_measure | M(A) indep M(B) for A cap B = empty, characterized by Levy-Ito decomp | estimate |
| Ghosal App K | gh_ap_k2 | ghosal_assouad_lemma | R_n >= (m/2) * min_{d(P0,P1)=1} d(P_tau0, P_tau1)^n / 4 | estimate |
| Ghosal App M | gh_ap_m1 | ghosal_mh_sampler | Accept theta* with prob min(1, pi(theta*\|X)*q(theta\|theta*) / (pi(theta\|X)*q(theta*\|theta))) | estimate |
| Ghosal App M | gh_ap_m2 | ghosal_gibbs_sampler | theta_j^{(t+1)} ~ pi(theta_j \| theta_{-j}^{(t)}, X) | estimate |
| Ghosal App M | gh_ap_m3 | ghosal_slice_sampler | Sample u ~ Unif(0, pi(theta\|X)), then theta ~ Unif({theta: pi(theta\|X)>u}) | estimate |
| Ghosal Ch 1 §1.3 | gh_c1_1 | ghosal_bayes_rule_infinite | pi(theta\|X) proportional to p_theta(X) * pi(theta) | estimate |
| Ghosal Ch 1 §1.3 | gh_c1_3 | ghosal_prior_posterior_update | dPi_n/dPi(theta) = p_theta^(n)(X^n) / integral p_eta^(n)(X^n) dPi(eta) | estimate |
| Ghosal Ch 1 §1.3.1 | gh_c1_2 | ghosal_absolute_continuity | P_theta << mu for all theta, posterior exists via Radon-Nikodym | estimate |
| Ghosal Ch 10 §10.1 | gh_c10_1 | ghosal_adapt_thm | Pi = sum_k pi_k * Pi_k, pi_k ~ exp(-lambda*k*log n), rate eps_n(f0) auto | estimate |
| Ghosal Ch 10 §10.2.1 | gh_c10_2 | ghosal_univ_weights | pi_k: sum_k exp(log pi_k + n*eps_k^2) converges for all eps_k | estimate |
| Ghosal Ch 10 §10.2.2 | gh_c10_3 | ghosal_param_rate | If f0 in parametric model with dim d, adaptation gives sqrt(d/n) rate | estimate |
| Ghosal Ch 10 §10.2.3 | gh_c10_4 | ghosal_two_model_adp | Pi = pi_0*Pi_0 + (1-pi_0)*Pi_1, posterior weights adapt to data complexity | estimate |
| Ghosal Ch 10 §10.3.2 | gh_besov_prior | ghosal_besov_prior | theta_{jk} ~ pi_j * N(0, 2^{-j(2s+1)}) + (1-pi_j)*delta_0 | estimate |
| Ghosal Ch 10 §10.3.2 | gh_c10_5 | ghosal_wn_adapt | theta_jk ~ pi*N(0,tau_j^2) + (1-pi)*delta_0, adapts to Besov smoothness | estimate |
| Ghosal Ch 10 §10.4 | gh_c10_6 | ghosal_rnd_series_pr | K ~ pi_n, beta_k\|K ~ N(0,sigma^2) iid, adaptive to smoothness s | estimate |
| Ghosal Ch 10 §10.4.2 | gh_c10_8 | ghosal_frs_reg | Y_i = f(x_i)+e_i, f = sum_{k<=K} beta_k phi_k, rate n^{-2s/(2s+1)} | estimate |
| Ghosal Ch 10 §10.4.3 | gh_c10_9 | ghosal_frs_binreg | P(Y=1\|x) = Phi(f(x)), f = sum_{k<=K} beta_k phi_k, adaptive rate | estimate |
| Ghosal Ch 10 §10.4.4 | gh_c10_10 | ghosal_frs_poireg | Y\|x ~ Poi(exp(f(x))), f = sum_{k<=K} beta_k phi_k, rate adaptation | estimate |
| Ghosal Ch 10 §10.4.5 | gh_c10_11 | ghosal_func_reg | E[Y\|X,T] = integral X(t) beta(t) dt, beta ~ series prior, adaptive | estimate |
| Ghosal Ch 10 §10.5 | gh_c10_12 | ghosal_modsel_bic | BF(H1,H0) = P(X^n\|H1)/P(X^n\|H0) -> infty if H1 true, -> 0 if H0 true | estimate |
| Ghosal Ch 10 §10.5.3 | gh_c10_14 | ghosal_param_np_bf | BF ~ exp(-n*KL(P0,Phat_MLE)) / Pi(KL ball eps_n) for parametric H0 | estimate |
| Ghosal Ch 11 §11.2 | gh_c11_1 | ghosal_gp_def_rkhs | H_k = closure{sum a_i k(x_i,.): a_i in R}, inner product <k(x,.),k(y,.)>_H=k(x,y) | estimate |
| Ghosal Ch 11 §11.3 | gh_c11_2 | ghosal_rkhs_norm | phi_n(eps) = inf_{h in H: \|\|h-f0\|\|<eps} \|\|h\|\|_H^2 - log Pi(\|\|f\|\|<eps) | estimate |
| Ghosal Ch 11 §11.3 | gh_c11_3 | ghosal_gp_crt_thm | eps_n satisfying n*eps_n^2 >= phi_n(eps_n) gives contraction rate | estimate |
| Ghosal Ch 11 §11.3 | gh_conc_func | ghosal_concentration_function | phi_{f0}(eps) = inf_{\|\|h-f0\|\|<eps} \|\|h\|\|_H^2 - log Pi(\|\|f-0\|\|<eps) | estimate |
| Ghosal Ch 11 §11.3 | gh_small_ball | ghosal_small_ball_prob | phi(eps) = log(1/Pi(\|\|f\|\|_infty < eps)) controls prior concentration | estimate |
| Ghosal Ch 11 §11.3.2 | gh_c11_5 | ghosal_gp_binreg_crt | f ~ GP(0,k), rate n^{-s/(2s+d/2)} for GP on R^d with Sobolev kernel | estimate |
| Ghosal Ch 11 §11.3.3 | gh_sup_norm_gp | ghosal_sup_norm_contraction | Pi_n(\|\|f-f0\|\|_infty > M*eps_n \| data) -> 0 at rate n^{-s/(2s+d)} * (log n)^t | estimate |
| Ghosal Ch 11 §11.4.1 | gh_c11_6 | ghosal_bm_prior | W ~ BM: E[W(t)]=0, Cov(W(s),W(t))=min(s,t), paths Holder-1/2 a.s. | estimate |
| Ghosal Ch 11 §11.4.1 | gh_gp_brow_prim | ghosal_gp_brownian_primitive | W^{(m)} = m-fold integrated BM, k_m covariance, paths C^{m-1} | estimate |
| Ghosal Ch 11 §11.4.2 | gh_c11_7 | ghosal_rl_process | R_s(t) = integral_0^t (t-u)^{s-1/2} dW(u) / Gamma(s+1/2) | estimate |
| Ghosal Ch 11 §11.4.3 | gh_c11_8 | ghosal_fbm_prior | fBM: k(s,t) = (\|s\|^{2H}+\|t\|^{2H}-\|s-t\|^{2H})/2, H in (0,1) | estimate |
| Ghosal Ch 11 §11.4.4 | gh_c11_9 | ghosal_statgp_spec | k(x,y) = integral exp(i*omega'*(x-y)) dF(omega) by Bochner theorem | estimate |
| Ghosal Ch 11 §11.4.4 | gh_gp_orn_uhl | ghosal_gp_ornstein_uhlenbeck | k(s,t) = sigma^2 exp(-\|s-t\|/l), stationary Markov GP | estimate |
| Ghosal Ch 11 §11.4.5 | gh_c11_10 | ghosal_series_gp | f ~ GP with k(x,y) = sum_k lambda_k phi_k(x) phi_k(y), eigenexpansion | estimate |
| Ghosal Ch 11 §11.5 | gh_c11_11 | ghosal_rescal_gp | f_l(x) = f(x/l), l ~ pi(l), adapts to unknown smoothness | estimate |
| Ghosal Ch 11 §11.5.1 | gh_c11_12 | ghosal_selfsim_gp | fBM: f(lambda*.) =_d lambda^H f(.) for Hurst H | estimate |
| Ghosal Ch 11 §11.6 | gh_c11_13 | ghosal_gp_adapt_thm | f ~ GP(0, k_{l_n}), l_n ~ Pi_l, rate adapts to any s-Holder f0 | estimate |
| Ghosal Ch 11 §11.7.4 | gh_c11_15 | ghosal_ep_gp | q(f) = N(mu, Sigma) with Sigma = (K^{-1} + sum Lambda_i)^{-1} | estimate |
| Ghosal Ch 11 §11.7.5 | gh_c11_14 | ghosal_gp_laplace | pi(f\|data) approx N(f_hat, (K^{-1}+W)^{-1}), W = diag(-nabla^2 log p(y\|f)) | estimate |
| Ghosal Ch 12 §12.1 | gh_c12_1 | ghosal_infdim_bvm | sqrt(n)(Pi_n - N(theta_hat, I_n^{-1})) -> 0 in total variation | estimate |
| Ghosal Ch 12 §12.2 | gh_c12_2 | ghosal_dp_bvm | sqrt(n)(G_n(.) - F_0(.)) -> B(F_0(.)) weakly, B = standard Brownian bridge | estimate |
| Ghosal Ch 12 §12.2.1 | gh_c12_3 | ghosal_strong_apx_dp | sup_t \|sqrt(n)(G_n(t) - F_0(t)) - B(F_0(t))\| -> 0 a.s. | estimate |
| Ghosal Ch 12 §12.3 | gh_c12_4 | ghosal_semipara_bvm | sqrt(n)(psi(G_n) - psi(F_0)) -> N(0, sigma_eff^2) with sigma_eff^2 = Cramer-Rao lb | estimate |
| Ghosal Ch 12 §12.3 | gh_c12_6 | ghosal_semipara_eff | var >= (nabla psi)^T I_{theta,eta}^{-1} nabla psi (semiparametric Cramer-Rao) | estimate |
| Ghosal Ch 12 §12.3 | gh_mises_eff | ghosal_mises_efficiency | psi(P) Mises-differentiable at P0 with influence function tilde{psi} => efficient BvM | estimate |
| Ghosal Ch 12 §12.3.1 | gh_c12_5 | ghosal_eff_infl_fn | psi(P) - psi(P0) = E_{P0}[tilde{psi}(X)] + o(\|\|P-P0\|\|), tilde{psi} = eff. inf. fn. | estimate |
| Ghosal Ch 12 §12.3.2 | gh_c12_7 | ghosal_strict_sbvm | Conditions on prior for exact semiparametric BvM to hold at true theta_0 | estimate |
| Ghosal Ch 12 §12.3.3 | gh_c12_8 | ghosal_cox_bvm_sp | sqrt(n)(beta_n - beta_0) -> N(0, I_beta^{-1}) via partial likelihood BvM | estimate |
| Ghosal Ch 12 §12.4.1 | gh_c12_9 | ghosal_wn_full_bvm | dY = theta dt + dW/sqrt(n), Pi_n -> N(Y, I/n) in total variation | estimate |
| Ghosal Ch 12 §12.4.2 | gh_c12_10 | ghosal_wn_lin_bvm | sqrt(n)(L(theta_n) - L(theta_0)) -> N(0, \|\|L\|\|^2) for linear functional L | estimate |
| Ghosal Ch 12 §12.5 | gh_c12_11 | ghosal_cred_set_cov | Pi_n(theta: \|\|theta-theta_0\|\|<=r_n \| X^n) -> 1-alpha => P0^n-prob -> 1-alpha | estimate |
| Ghosal Ch 12 §12.5 | gh_inf_dim_cr | ghosal_inf_dim_credible | Pi_n(\|\|theta - theta0\|\|_H <= r_n \| X^n) -> 1-alpha in P0^n probability | estimate |
| Ghosal Ch 13 §13.2 | gh_c13_1 | ghosal_surv_dp_post | F\|X ~ DP posterior accounting for censored observations | estimate |
| Ghosal Ch 13 §13.3 | gh_c13_3 | ghosal_beta_proc_def | BP(c, H0): increments dH(t) ~ Be(c(t)*dH0(t), c(t)*(1-dH0(t))) ind. | estimate |
| Ghosal Ch 13 §13.3.1 | gh_c13_4 | ghosal_bp_discrete | H(t) = sum_{s<=t} dH(s), dH(s_k) ~ Be(c_k*h_k, c_k*(1-h_k)) independent | estimate |
| Ghosal Ch 13 §13.3.2 | gh_c13_5 | ghosal_bp_cont | BP(c,H0): Levy measure nu(du x dt) = u^{-1}(1-u)^{c(t)-1} du * c(t) dH0(t) | estimate |
| Ghosal Ch 13 §13.3.3 | gh_c13_6 | ghosal_bp_path_gen | H(t) = sum_{tau_k <= t} J_k, (J_k, tau_k) from Poisson process on [0,1]x[0,T] | estimate |
| Ghosal Ch 13 §13.3.4 | gh_c13_7 | ghosal_mix_bp | H ~ integral BP(c, H0_lambda) dPi(lambda) | estimate |
| Ghosal Ch 13 §13.4 | gh_c13_8 | ghosal_ntr_def | F(t) = 1 - exp(-integral_0^t log(1-u) dN(u,s)), M ~ NTR Levy process | estimate |
| Ghosal Ch 13 §13.4 | gh_c13_9 | ghosal_ntr_levy | Laplace functional: E[exp(-integral f dM)] = exp(-integral (1-e^{-f}) dnu) | estimate |
| Ghosal Ch 13 §13.4.1 | gh_c13_10 | ghosal_ntr_consist | Pi_n(F: d(F,F0)>eps \| X^n) -> 0 under KL support condition on NTR | estimate |
| Ghosal Ch 13 §13.4.2 | gh_c13_11 | ghosal_ntr_bvm | sqrt(n)(psi(F_n) - psi(F_0)) -> N(0, sigma^2) for smooth functional psi | estimate |
| Ghosal Ch 13 §13.5 | gh_c13_12 | ghosal_smhaz_gp | lambda(t) = exp(f(t)), f ~ GP(mu, k), hazard smooth via GP | estimate |
| Ghosal Ch 13 §13.6 | gh_c13_13 | ghosal_cox_model | lambda(t\|x) = lambda_0(t) exp(beta'x), lambda_0 ~ BP, beta ~ Normal | estimate |
| Ghosal Ch 13 §13.6.1 | gh_c13_14 | ghosal_cox_post | pi(beta \| data) propto pi(beta) * prod exp(beta'x_i - log sum_{j in R_i} exp(beta'x_j)) | estimate |
| Ghosal Ch 13 §13.7.1 | gh_c13_16 | ghosal_bb_censored | DP posterior at alpha->0 for censored data gives Lo's estimator | estimate |
| Ghosal Ch 14 §14.1 | gh_c14_1 | ghosal_eppf_def | P(partition n into k blocks of sizes n_1..n_k) = p(n_1..n_k) symmetric | estimate |
| Ghosal Ch 14 §14.1 | gh_c14_2 | ghosal_ewens_esf | p(n_1..n_k) = alpha^k * prod_{j=1}^k (n_j-1)! / prod_{i=0}^{n-1}(alpha+i) | estimate |
| Ghosal Ch 14 §14.1.1 | gh_c14_3 | ghosal_crp_def | P(C_{n+1}=k \| C_1..C_n) = n_k/(alpha+n) or alpha/(alpha+n) new table | estimate |
| Ghosal Ch 14 §14.1.2 | gh_c14_4 | ghosal_crf_def | CRF: global menu G0, per-restaurant G_j ~ DP(alpha, G0), sharing dishes | estimate |
| Ghosal Ch 14 §14.10 | gh_c14_23 | ghosal_ibp_def | Z in {0,1}^{n x K}: P(feature k assigned to customer i) from IBP | estimate |
| Ghosal Ch 14 §14.10 | gh_c14_24 | ghosal_ibp_stickbr | pi_k = prod_{j=1}^k V_j, V_j iid Beta(alpha,1), P(Z_{ik}=1)=pi_k | estimate |
| Ghosal Ch 14 §14.10 | gh_c14_25 | ghosal_ibp_poisson | Features = Poisson process on [0,1] with intensity alpha, thinned per customer | estimate |
| Ghosal Ch 14 §14.2 | gh_c14_5 | ghosal_ssp_def | G = sum_{k=1}^infty p_k delta_{theta_k}, theta_k iid G0, sum p_k=1 a.s. | estimate |
| Ghosal Ch 14 §14.2.1 | gh_c14_6 | ghosal_ssp_post | G\|X_1..X_n from SSP with updated weights for observed species | estimate |
| Ghosal Ch 14 §14.2.2 | gh_c14_7 | ghosal_ssp_mix | f(x) = integral K(x;theta) dG(theta), G = sum p_k delta_{theta_k} ~ SSP | estimate |
| Ghosal Ch 14 §14.3 | gh_c14_8 | ghosal_gibbs_proc | p(n_1..n_k) = C_n * prod_{j=1}^k V_{n_j} for Gibbs-type process | estimate |
| Ghosal Ch 14 §14.4 | gh_c14_10 | ghosal_py_eppf | p(n_1..n_k) = prod_{j=1}^{k-1}(theta+j*d) / prod_{i=1}^{n-1}(theta+i) * prod_{j=1}^k prod_{l=0}^{n_j | estimate |
| Ghosal Ch 14 §14.4 | gh_c14_11 | ghosal_py_powerlaw | E[K_n] ~ Gamma(theta+1)/Gamma(theta+d) * n^d / Gamma(1-d) | estimate |
| Ghosal Ch 14 §14.4 | gh_c14_9 | ghosal_py_process | V_k ~ Beta(1-d, theta+k*d), d in [0,1), theta > -d | estimate |
| Ghosal Ch 14 §14.4 | gh_py_univ_seq | ghosal_py_universal_sequence | (V_k) iid Beta(1-d, theta+k*d) for PY(d,theta,G0), V_0=1 | estimate |
| Ghosal Ch 14 §14.5 | gh_c14_12 | ghosal_pk_process | G = sum_k (J_k/T) delta_{theta_k}, J_k from Poisson process with Levy measure rho | estimate |
| Ghosal Ch 14 §14.5 | gh_c14_13 | ghosal_pk_levy | rho(du): Levy measure on (0,infty), G = normalized sum of Poisson jumps | estimate |
| Ghosal Ch 14 §14.6 | gh_c14_14 | ghosal_nig_proc | G = sum_k J_k/T delta_{theta_k}, J_k from IG Levy: rho(du)=exp(-alpha^2*u/2)/sqrt(2*pi*u^3) du | estimate |
| Ghosal Ch 14 §14.7 | gh_c14_15 | ghosal_ncrm_def | G(A) = M(A)/M(X), M ~ CRM: M(A) = sum_{tau_k in A} J_k | estimate |
| Ghosal Ch 14 §14.7 | gh_c14_16 | ghosal_ncrm_levy | E[exp(-integral f dM)] = exp(-integral (1-e^{-f(x)*u}) nu(du,dx)) | estimate |
| Ghosal Ch 14 §14.8 | gh_c14_17 | ghosal_disc_rp_rel | DP subset PY subset PK subset NCRM, each is special case of next | estimate |
| Ghosal Ch 14 §14.9.1 | gh_c14_18 | ghosal_ksbp_def | p_k(x) = V_k(x) prod_{j<k}(1-V_j(x)), V_k(x) = g(w_k'x) | estimate |
| Ghosal Ch 14 §14.9.2 | gh_c14_19 | ghosal_local_dp | G(x,.) = sum_k w_k(x) delta_{theta_k}, w_k(x) from kernel stick-breaking | estimate |
| Ghosal Ch 14 §14.9.2 | gh_loc_dp_crt | ghosal_local_dp_rate | Local DP(alpha(x), G0): rate n^{-s/(2s+1)} * (log n)^t for regression | estimate |
| Ghosal Ch 14 §14.9.3 | gh_c14_20 | ghosal_probit_sbp | G(x,.) = sum_k w_k(x) delta_{theta_k}, w_k(x) via Phi(linear predictor) | estimate |
| Ghosal Ch 14 §14.9.4 | gh_c14_21 | ghosal_ord_dep_sbp | V_k = V_k(x_{(k)}), ordering-dependent weights for regression | estimate |
| Ghosal Ch 14 §14.9.5 | gh_c14_22 | ghosal_nested_dp | G_j \| G0 ~ DP(alpha, G0), G0 ~ DP(gamma, H) | estimate |
| Ghosal Ch 2 §2.1 | gh_c2_1 | ghosal_random_basis_expansion | f = sum_{k=1}^infty z_k phi_k, z_k ~ pi_k independently | estimate |
| Ghosal Ch 2 §2.2.1 | gh_c2_2 | ghosal_gp_prior_def | f ~ GP(mu, k): E[f(x)]=mu(x), Cov(f(x),f(x'))=k(x,x') | estimate |
| Ghosal Ch 2 §2.2.2 | gh_c2_3 | ghosal_gp_increasing_prior | F(t) = integral_0^t exp(W(s)) ds, W ~ GP | estimate |
| Ghosal Ch 2 §2.3.1 | gh_c2_4 | ghosal_exp_link | f(x) = exp(psi(x)) / integral exp(psi(t)) dt | estimate |
| Ghosal Ch 2 §2.3.2 | gh_c2_5 | ghosal_histogram_prior | f(x) = sum_k p_k / \|B_k\| * 1_{x in B_k}, (p_1..p_K) ~ Dir(alpha) | estimate |
| Ghosal Ch 2 §2.3.3 | gh_c2_6 | ghosal_mixture_basis_prior | f = sum_k w_k K(x; theta_k), (w_k) ~ Dirichlet | estimate |
| Ghosal Ch 2 §2.3.4 | gh_c2_7 | ghosal_bernstein_feller | F_K(x) = sum_{k=0}^K F(k/K) C(K,k) x^k (1-x)^{K-k} | estimate |
| Ghosal Ch 2 §2.4 | gh_c2_8 | ghosal_np_normal_reg | Y_i = f(x_i) + e_i, e_i ~ N(0,sigma^2), f ~ GP(0, k) | estimate |
| Ghosal Ch 2 §2.5 | gh_c2_9 | ghosal_np_binary_reg | P(Y=1\|x) = Phi(f(x)), f ~ GP prior | estimate |
| Ghosal Ch 2 §2.6 | gh_c2_10 | ghosal_np_poisson_reg | Y\|x ~ Poisson(exp(f(x))), f ~ GP prior | estimate |
| Ghosal Ch 3 §3.1 | gh_c3_1 | ghosal_random_measure_def | G: Omega -> M(X), consistent finite-dimensional distributions via Kolmogorov | estimate |
| Ghosal Ch 3 §3.2 | gh_c3_2 | ghosal_stochastic_proc_prior | G(A_1..A_k) ~ p(v_1..v_k) with consistency conditions | estimate |
| Ghosal Ch 3 §3.3.1 | gh_c3_3 | ghosal_dir_simplex | (p_1..p_k) ~ Dir(alpha_1..alpha_k): p_j = G_j/sum G_i, G_j ~ Ga(alpha_j,1) | estimate |
| Ghosal Ch 3 §3.3.2 | gh_c3_4 | ghosal_stick_break_def | G = sum_k w_k delta_{theta_k}, V_k ~ Beta(a_k, b_k), theta_k ~ G0 | estimate |
| Ghosal Ch 3 §3.3.3 | gh_c3_5 | ghosal_countable_dp | (G(x_1), G(x_2), ...) ~ Dir(alpha*G0(x_1), alpha*G0(x_2), ...) | estimate |
| Ghosal Ch 3 §3.4.1 | gh_c3_6 | ghosal_dense_subset_prior | G = sum_k w_k delta_{X_k}, X_k from dense sequence | estimate |
| Ghosal Ch 3 §3.4.3 | gh_c3_7 | ghosal_rect_partition | G random measure constructed via random rectangular cells | estimate |
| Ghosal Ch 3 §3.4.5 | gh_c3_9 | ghosal_quantile_prior | G via Q(u) = F^{-1}(u), u in [0,1] | estimate |
| Ghosal Ch 3 §3.4.6 | gh_c3_10 | ghosal_norm_crm | G(A) = M(A)/M(X), M ~ CRM with Levy measure nu | estimate |
| Ghosal Ch 3 §3.6 | gh_c3_11 | ghosal_tailfree_def | (G(B_e0)/G(B_e), G(B_e1)/G(B_e)) independent across partitions | estimate |
| Ghosal Ch 3 §3.7 | gh_c3_12 | ghosal_polya_tree_def | PT(T_m, A): Y_{e0\|e} ~ Beta(alpha_{e0}, alpha_{e1}), G(B_e0)=Y_{e0\|e}*G(B_e) | estimate |
| Ghosal Ch 3 §3.7.1 | gh_c3_13 | ghosal_polya_urn_pt | P(X_{n+1} in B \| X_1..X_n) = (alpha_B + n_B) / (alpha + n) | estimate |
| Ghosal Ch 3 §3.7.3 | gh_c3_15 | ghosal_partspec_pt | PT specified only at selected levels m1 < m2 < ..., rest marginalized | estimate |
| Ghosal Ch 3 §3.7.4 | gh_c3_16 | ghosal_evsplit_pt | PT*(alpha, a_m): alpha_{e0}=alpha_{e1}=a_m at level m | estimate |
| Ghosal Ch 4 §4.1 | gh_c4_1 | ghosal_dp_def | (G(A_1)..G(A_k)) ~ Dir(alpha*G0(A_1)..alpha*G0(A_k)) for any partition | estimate |
| Ghosal Ch 4 §4.1.1 | gh_c4_2 | ghosal_dp_mean | E[G(A)] = G0(A) | estimate |
| Ghosal Ch 4 §4.1.1 | gh_c4_3 | ghosal_dp_var | Var[G(A)] = G0(A)(1-G0(A)) / (alpha+1) | estimate |
| Ghosal Ch 4 §4.1.1 | gh_c4_4 | ghosal_dp_cov | Cov[G(A),G(B)] = (G0(A cap B) - G0(A)G0(B)) / (alpha+1) | estimate |
| Ghosal Ch 4 §4.1.2 | gh_c4_5 | ghosal_dp_selfsim | G\|G(A)=w ~ w*DP(alpha*G0(.\|A)) + (1-w)*DP(alpha*G0(.\|A^c)) | estimate |
| Ghosal Ch 4 §4.1.2 | gh_dp_cond_dist | ghosal_dp_conditional_distribution | G(A^c . \| G(A)=w) \| G(A)=w ~ DP(alpha*G0(A^c/(1-G0(A))), *) on A^c | estimate |
| Ghosal Ch 4 §4.1.3 | gh_c4_6 | ghosal_dp_post | G \| X_1..X_n ~ DP(alpha+n, (alpha*G0 + sum delta_{X_i})/(alpha+n)) | estimate |
| Ghosal Ch 4 §4.1.4 | gh_c4_7 | ghosal_dp_pred | X_{n+1}\|X_1..X_n ~ (alpha*G0 + sum delta_{X_i})/(alpha+n) | estimate |
| Ghosal Ch 4 §4.1.5 | gh_c4_8 | ghosal_dp_ndist | E[K_n] = sum_{i=1}^n alpha/(alpha+i-1) ~ alpha*log(n) | estimate |
| Ghosal Ch 4 §4.2.3 | gh_c4_9 | ghosal_dp_gamma | G(A) = Gamma_A / Gamma_X, Gamma_A ~ Ga(alpha*G0(A),1) | estimate |
| Ghosal Ch 4 §4.2.4 | gh_c4_10 | ghosal_dp_polya_urn | X_{n+1}\|X_1..X_n ~ sum_k n_k/(alpha+n)*delta_{X_k^*} + alpha/(alpha+n)*G0 | estimate |
| Ghosal Ch 4 §4.2.5 | gh_c4_11 | ghosal_dp_stickbr | w_k = V_k prod_{j<k}(1-V_j), V_k iid Beta(1,alpha), theta_k iid G0 | estimate |
| Ghosal Ch 4 §4.3.1 | gh_c4_12 | ghosal_dp_discrete | G ~ DP(alpha,G0) => G discrete a.s., supp(G) subset supp(G0) a.s. | estimate |
| Ghosal Ch 4 §4.3.2 | gh_c4_13 | ghosal_dp_weak_conv | alpha_n -> alpha, G0_n ->_w G0 => DP(alpha_n,G0_n) ->_w DP(alpha,G0) | estimate |
| Ghosal Ch 4 §4.3.3 | gh_c4_14 | ghosal_dp_fs_approx | G_K = sum_{k=1}^K w_k delta_{theta_k} + w_{K+1} G0, approx error bounded | estimate |
| Ghosal Ch 4 §4.3.4 | gh_c4_15 | ghosal_dp_mutual_sing | DP(alpha, G0) perp DP(alpha', G0') if G0 != G0' | estimate |
| Ghosal Ch 4 §4.3.5 | gh_c4_16 | ghosal_dp_tails | P(G(A) > t) <= exp(-C*alpha*t) for t > G0(A) | estimate |
| Ghosal Ch 4 §4.3.6 | gh_c4_17 | ghosal_dp_median | Med(G) ~ distribution determined by DP(alpha, G0) via quantile inversion | estimate |
| Ghosal Ch 4 §4.3.7 | gh_c4_18 | ghosal_dp_mean_dist | E[G] = integral x dG(x), distribution via Cifarelli-Regazzini formula | estimate |
| Ghosal Ch 4 §4.4 | gh_c4_19 | ghosal_dp_charact | G ~ DP iff all finite marginals are Dirichlet (Ferguson) or stick-breaking (Sethuraman) | estimate |
| Ghosal Ch 4 §4.5 | gh_c4_20 | ghosal_mix_dp | G \| G0 ~ DP(alpha, G0), G0 ~ H => marginal is mixture of DP | estimate |
| Ghosal Ch 4 §4.5 | gh_hier_np | ghosal_hierarchical_np | theta_i\|G ~ G, G\|alpha ~ DP(alpha,G0), alpha ~ Ga(a,b) | estimate |
| Ghosal Ch 4 §4.6.1 | gh_c4_21 | ghosal_inv_dp | IDP(alpha): G =_d T#G for transformation T in group G | estimate |
| Ghosal Ch 4 §4.6.2 | gh_c4_22 | ghosal_constr_dp | DP(alpha, G0) \| integral f dG = c | estimate |
| Ghosal Ch 4 §4.6.3 | gh_c4_23 | ghosal_pen_dp | pi(G) propto exp(-lambda*d(G,G0)) * DP(alpha,G0)(G) | estimate |
| Ghosal Ch 4 §4.7 | gh_c4_24 | ghosal_bayes_boot | G \| X_1..X_n ~ DP(epsilon, G_n) as epsilon->0 gives Bayesian bootstrap | estimate |
| Ghosal Ch 5 §5.1 | gh_c5_1 | ghosal_dpm_model | f(x) = integral K(x;theta) dG(theta), G ~ DP(alpha,G0) | estimate |
| Ghosal Ch 5 §5.1 | gh_c5_2 | ghosal_dpm_marg | p(X_1..X_n) = prod_i p(X_i \| X_1..X_{i-1}) via Polya urn predictive | estimate |
| Ghosal Ch 5 §5.2 | gh_c5_3 | ghosal_cgibbs | p(c_i=k\|rest) proportional to n_{-i,k}*f(X_i\|theta_k) or alpha*p_0(X_i) | estimate |
| Ghosal Ch 5 §5.2 | gh_c5_4 | ghosal_splitmerge | Split: divide cluster k into two; Merge: join clusters k,l into one via MH | estimate |
| Ghosal Ch 5 §5.2 | gh_c5_5 | ghosal_blk_gibbs | G_K = sum_{k=1}^K w_k delta_{theta_k}, update all K components jointly | estimate |
| Ghosal Ch 5 §5.3 | gh_c5_6 | ghosal_vb_dpm | ELBO: E_q[log p(X,z,theta,G)] - E_q[log q(z,theta,G)] maximized | estimate |
| Ghosal Ch 5 §5.3 | gh_var_dp_post | ghosal_variational_dp_posterior | q*(G_K, theta, z) = argmin KL(q\|\|pi(.\|X)), truncation at K | estimate |
| Ghosal Ch 5 §5.5 | gh_c5_10 | ghosal_poi_ker | f(k) = integral Poi(k;lambda) dG(lambda), G ~ DP | estimate |
| Ghosal Ch 6 §6.1 | gh_c6_1 | ghosal_weak_consist | Pi_n({P: d_w(P,P0) > eps} \| X_1..X_n) ->_{P0} 0 | estimate |
| Ghosal Ch 6 §6.1 | gh_c6_2 | ghosal_strong_consist | Pi_n({P: d(P,P0) > eps} \| X_1..X_n) -> 0 P0^infty-a.s. | estimate |
| Ghosal Ch 6 §6.2 | gh_c6_3 | ghosal_doob_consist | Pi-a.s., Pi_n(U^c\|X^n) -> 0 for all U with P_theta in U | estimate |
| Ghosal Ch 6 §6.3 | gh_c6_4 | ghosal_df_inconsist | Counterexample: Pi consistent only on Pi-null set of theta_0 | estimate |
| Ghosal Ch 6 §6.4 | gh_c6_6 | ghosal_kl_support | Pi({P: KL(P0,P)<eps}) > 0 for all eps > 0 | estimate |
| Ghosal Ch 6 §6.4 | gh_c6_7 | ghosal_kl_diverge | KL(P0,P) = integral log(p0/p) dP0 | estimate |
| Ghosal Ch 6 §6.4 | gh_iid_consist | ghosal_iid_posterior_consistency | P0^infty: Pi_n(d_w(P,P0)>eps \| X^n) -> 0 under Schwartz conditions | estimate |
| Ghosal Ch 6 §6.5 | gh_c6_8 | ghosal_tailfree_con | PT(T_m, A) with alpha_m -> infty has KL support at any continuous P0 | estimate |
| Ghosal Ch 6 §6.6 | gh_c6_9 | ghosal_kl_perm | If Pi1, Pi2 have KL property at P0, so does their mixture and product | estimate |
| Ghosal Ch 6 §6.7.1 | gh_c6_10 | ghosal_non_iid_con | Pi_n(U^c\|X^n)->0 under triangular array conditions on p_{n,theta} | estimate |
| Ghosal Ch 6 §6.7.2 | gh_c6_11 | ghosal_markov_con | KL(P0^Markov, P^Markov) controlled by stationary KL divergence | estimate |
| Ghosal Ch 6 §6.8.2 | gh_c6_13 | ghosal_lecam_consist | Pi_n(U^c\|X^n) <= (1/alpha_n) sum p(X_i\|theta)/p0(X_i) for test phi_n | estimate |
| Ghosal Ch 6 §6.8.3 | gh_c6_14 | ghosal_pred_consist | sum_{i=1}^n KL(P0, Pi-predictive at i) / n -> 0 a.s. | estimate |
| Ghosal Ch 6 §6.8.5 | gh_c6_16 | ghosal_alpha_post | pi_alpha(theta\|X^n) proportional pi(theta) * L_n(theta)^alpha, 0<alpha<1 | estimate |
| Ghosal Ch 7 §7.1.1 | gh_c7_1 | ghosal_pt_kl_prop | PT*(alpha, a_m) with a_m = alpha_m^2 satisfies KL condition | estimate |
| Ghosal Ch 7 §7.1.2 | gh_c7_2 | ghosal_kern_mix_kl | KL(p0, f_G) controlled by KL(G_n, G) for approximating measure G_n | estimate |
| Ghosal Ch 7 §7.1.3 | gh_c7_3 | ghosal_exp_dens_kl | pi(psi) = exp(-lambda\|\|psi\|\|) for psi in Sobolev ball has KL support | estimate |
| Ghosal Ch 7 §7.2 | gh_dp_kl_nbhd | ghosal_dp_kl_nbhd_mass | Pi(KL(P0,P)<eps) >= exp(-C*eps^{-1/s}) for DP with smooth G0 | estimate |
| Ghosal Ch 7 §7.2.2 | gh_c7_5 | ghosal_dpm_gen_con | DPM with kernel K satisfying tail and smoothness conditions is consistent | estimate |
| Ghosal Ch 7 §7.2.3 | gh_ppt_consist | ghosal_polya_tree_consist_rate | PT*(alpha, m^2): contraction rate n^{-s/(2s+1)} in Hellinger | estimate |
| Ghosal Ch 7 §7.3.2 | gh_dp_reg_post | ghosal_dp_regression_posterior | Y = f(X) + e, e ~ G ~ DP(alpha, G0), posterior for (f, G) | estimate |
| Ghosal Ch 7 §7.4.1 | gh_c7_8 | ghosal_loc_semipara | Y_i = theta + e_i, e_i ~ f, both theta and f estimated | estimate |
| Ghosal Ch 7 §7.4.2 | gh_c7_9 | ghosal_linreg_unk_err | Y = X'beta + e, e ~ f unknown, joint consistency for (beta, f) | estimate |
| Ghosal Ch 7 §7.4.3 | gh_c7_10 | ghosal_mono_reg_con | P(Y=1\|x) = F(x) monotone, DP prior on F, consistent in weak topology | estimate |
| Ghosal Ch 8 §8.1 | gh_c8_1 | ghosal_crt_def | Pi_n({theta: d(theta,theta0)>M*eps_n}\|X^n) ->_{P0^n} 0 for M->infty | estimate |
| Ghosal Ch 8 §8.2 | gh_c8_2 | ghosal_ggv_thm | eps_n contraction if: test condition + Pi(KL ball)>=exp(-n*eps_n^2) + entropy<=exp(n*eps_n^2) | estimate |
| Ghosal Ch 8 §8.2 | gh_c8_4 | ghosal_prior_mass_cnd | Pi({P: KL(P0,P)<eps_n^2, V2(P0,P)<eps_n^2}) >= exp(-n*eps_n^2) | estimate |
| Ghosal Ch 8 §8.2 | gh_c8_5 | ghosal_entropy_cnd | log N(eps_n, Theta_n, d_H) <= n*eps_n^2 for sieve Theta_n | estimate |
| Ghosal Ch 8 §8.2 | gh_contr_rate2 | ghosal_contraction_rate_iid | Pi_n(\|\|p-p0\|\|_1 > M*eps_n \| X^n) -> 0 if entropy and prior mass conditions hold | estimate |
| Ghosal Ch 8 §8.2.2 | gh_c8_7 | ghosal_fin_apx_pri | Pi_n = Pi on Theta_n, Theta_n = eps_n-net of size exp(n*eps_n^2) | estimate |
| Ghosal Ch 8 §8.3.2 | gh_c8_8 | ghosal_gauss_reg_crt | eps_n = sqrt(log n / n) for s-smooth regression function with s-smooth GP prior | estimate |
| Ghosal Ch 8 §8.3.3 | gh_c8_9 | ghosal_markov_crt | eps_n from stationary-distribution entropy and prior mass for Markov transitions | estimate |
| Ghosal Ch 8 §8.3.4 | gh_c8_10 | ghosal_wn_crt | dY_t = f(t)dt + n^{-1/2}dW_t, eps_n = n^{-s/(2s+1)} for s-Sobolev f | estimate |
| Ghosal Ch 8 §8.3.4 | gh_wn_rate_opt | ghosal_white_noise_optimal_rate | White noise dY = theta dt + dW/sqrt(n): minimax rate R_n* = n^{-2s/(2s+1)} | estimate |
| Ghosal Ch 8 §8.3.5 | gh_c8_11 | ghosal_ts_crt | eps_n rate for spectral density f via Whittle likelihood | estimate |
| Ghosal Ch 8 §8.4 | gh_c8_12 | ghosal_crt_lower | eps_n >= n^{-s/(2s+1)} for s-Holder smoothness (information-theoretic lb) | estimate |
| Ghosal Ch 8 §8.5 | gh_c8_13 | ghosal_misspec_crt | P* = argmin_{P in model} KL(P0,P), Pi_n({d(P,P*)>eps_n}\|X^n)->0 | estimate |
| Ghosal Ch 8 §8.5.1 | gh_c8_14 | ghosal_convex_misp | Model convex => P* unique, contraction at standard eps_n rate | estimate |
| Ghosal Ch 8 §8.6 | gh_c8_15 | ghosal_alpha_pst_crt | pi_alpha posterior contracts at same rate eps_n under weaker conditions | estimate |
| Ghosal Ch 9 §9.1 | gh_c9_1 | ghosal_logspline_crt | f = exp(sum beta_k phi_k)/Z, K_n ~ n^{1/(2s+1)}, rate n^{-s/(2s+1)} | estimate |
| Ghosal Ch 9 §9.2 | gh_c9_2 | ghosal_dp_disc_crt | DP prior: rate n^{-s/(2s+1)} * (log n)^t for density estimation | estimate |
| Ghosal Ch 9 §9.3 | gh_c9_3 | ghosal_bpoly_crt | f = sum_{k=0}^K p_k Be(x;k+1,K-k+1), K ~ pi_K, rate n^{-s/(2s+1)} | estimate |
| Ghosal Ch 9 §9.4 | gh_c9_4 | ghosal_dpm_norm_crt | DPM of N(mu,sigma^2): rate n^{-s/(2s+1)} * (log n)^t for s-smooth p0 | estimate |
| Ghosal Ch 9 §9.4.1 | gh_c9_5 | ghosal_norm_mix_apx | p0 in Sobolev(s) => exists G: \|\|p0 - integral phi_sigma dG\|\|_1 <= sigma^s | estimate |
| Ghosal Ch 9 §9.4.5 | gh_c9_6 | ghosal_wishart_dpm | Sigma ~ Wishart(nu, Psi), location-scale DPM, same rate as diagonal | estimate |
| Ghosal Ch 9 §9.5.2 | gh_c9_7 | ghosal_whittle_crt | Whittle likelihood rate: eps_n = n^{-s/(2s+1)} (log n)^(1/2) for s-smooth spectral f | estimate |
| Ghosal Ch 9 §9.5.3 | gh_c9_8 | ghosal_nlar_crt | X_t = f(X_{t-1}) + e_t, f ~ GP, rate n^{-s/(2s+1)} with ergodicity | estimate |
| Ghosal Ch 9 §9.5.4 | gh_c9_9 | ghosal_wn_conj_crt | dY = theta*dt + dW/sqrt(n), Gaussian prior => exact rate and distribution | estimate |
| Ghosal Ch 9 §9.5.4 | gh_sobol_prior | ghosal_sobolev_prior | theta_j ~ N(0, j^{-2s-1}), Sobolev(s) prior, rate n^{-2s/(2s+1)} | estimate |
| Ghosal Ch 9 §9.5.4 | gh_wn_gauss_pr | ghosal_white_noise_gauss_prior | dY = theta dt + dW/sqrt(n), theta ~ GP(0,C) => theta\|Y ~ N(posterior mean, posterior covariance) | estimate |
| Ghosal Ch 9 §9.5.5 | gh_c9_10 | ghosal_spline_crt | f in spline space of dim K_n, K_n ~ n^{1/(2s+1)}, rate n^{-2s/(2s+1)} | estimate |
| Ghosal Ch 9 §9.5.7 | gh_c9_11 | ghosal_icens_dp_crt | DP prior on monotone F for interval-censored data, rate (n/log n)^{-1/3} | estimate |
