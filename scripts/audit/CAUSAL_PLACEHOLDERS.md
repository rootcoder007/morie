# Causal placeholder worklist (119 named modules)

Triaged 2026-07-27 from the 5,502 named placeholders (census.py).
Method: implement in clusters sharing mathematics; verify every
citation against the library PDFs or the actual paper before use.

## Done

- `evalu` `evaltw` `causfromle` `ucbias` -- E-value cluster
  (VanderWeele & Ding 2017, doi:10.7326/M16-2607, verified;
  Ding & VanderWeele 2016)
- `jntmed` -- joint-significance max-p test (MacKinnon et al. 2002)

## Cluster plan (next)

1. **Mediation core** (bkmed/abind machinery exists): `causmedb`
   `mdian` (delegate to bkmed), `medfm` (Pearl mediation formula,
   discrete), `rhomed` `sensIM` (Imai-Keele-Yamamoto 2010 -- get the
   ACME(rho) formula from the paper, NOT from memory), `weakid`
2. **IPW / propensity** (aiptdd machinery): `causipsw` `msmest`
   `causmtch` `cipsc` `prsmtd` `unitnr` `nonresp` `spwgts` `causqte`
   (Firpo 2007)
3. **G-formula**: `gforml` `causmrop` `pluginM` `gctvc` `gmccsm`
4. **Synthetic control**: `caussc`/`scmaba` (one impl + front-end),
   `ascmcl` `causscg`/`gscmcl` `causscss`
5. **DiD**: `causdid2` (trivial 2x2), `drbqs` `drovrl` (extend aiptdd)
6. **Graph/identification checkers** (bdcrt machinery): `backDR`
   `fdadj` `fdcrt` `medFront` `cmark` `exchg` `ivcrt` `frkst` `chstr`
   `ident` `scmdf` `ctcfl` `potef`
7. **Granger / info-flow**: `ggrcst` (VAR F-test), `granci` `trnfen`
8. **Forest/TMLE tier** (heavy; design before implementing):
   `crfath` `crfboot` `crfhte` `csfgrf` `csurv2` `qbcfgr` `htgcrf`
   `drlnr` `tml*` (9 modules) `dml*` -- decide native scope first

## Remaining modules

- `abdpd` -- Three-step counterfactual inference: abduction, modification, prediction
- `ascmcl` -- Augmented Synthetic Control (Ben-Michael-Feller-Rothstein)
- `baymed` -- Bayesian mediation
- `bnscrd` -- Bound on causal effect under missing RD
- `causdid2` -- Canonical 2x2 difference-in-differences
- `causdmliv` -- Double ML for instrumental variables
- `causftbl` -- Pearl front-door adjustment via mediator path
- `causipsw` -- ATT inverse probability of treatment weights
- `causmedb` -- Baron-Kenny three-equation mediation
- `causmedi` -- Imai-Keele-Tingley sequential ignorability mediation
- `causmtch` -- Propensity-score nearest-neighbour 1:1 matching
- `causqte` -- Quantile treatment effect via Firpo IPW
- `causrho` -- Proximal causal inference via proxy bridge function
- `caussc` -- Abadie-Diamond-Hainmueller synthetic control weights
- `causscg` -- Generalised synthetic control via interactive fixed effects
- `causscss` -- Synthetic control subset selection (LASSO-relaxed)
- `causshap` -- Shapley value-based causal contribution decomposition
- `chstr` -- Chain (mediation) structure A->B->C: information flows, B is mediator
- `cipsc` -- Propensity score caliper matching (restrict to within-caliper pairs)
- `clstcr` -- Cluster-level causal inference
- `cmark` -- Causal Markov condition: each node independent of non-descendants given parents
- `counRS` -- Counterfactual rec evaluation
- `countMd` -- Mediation for count outcome
- `crfath` -- Causal forest (Wager-Athey) for heterogeneous treatment effects
- `crfboot` -- Bootstrap-honest causal forest
- `crfhte` -- Test for treatment effect heterogeneity (BLP statistic)
- `csfgrf` -- Causal survival forest for heterogeneous time-to-event treatment effects
- `csurv2` -- Best linear predictor for causal survival forest CATE
- `ctcfl` -- Counterfactual notation: Y_x outcome had X been set to x by intervention
- `ddrbnd` -- DR-bounds for instrumental variable LATE under monotonicity
- `deciA` -- DECI (deep end-to-end causal inference): joint structure + effect learning
- `dmlMed` -- Double ML mediation Neyman-orthogonal
- `drbqs` -- DR-DiD quantile treatment effect
- `drovrl` -- DR-DiD with propensity overlap trimming
- `exchg` -- Exchangeability (unconfoundedness/ignorability): treatment independent of potential outcomes given covariates
- `fciag` -- Fast Causal Inference (FCI) algorithm for hidden confounders
- `fdadj` -- Front-door adjustment formula for causal effect via mediator
- `fdcrt` -- Front-door criterion: identification via mediator when backdoor blocked
- `frkst` -- Fork (common cause) structure A<-B->C: B is confounder
- `gb1251` -- Kendall partial tau for controlling confounder z
- `gctvc` -- G-computation (parametric g-formula) for time-varying confounding
- `gforml` -- Robins g-formula -- Monte Carlo simulation of counterfactual outcome distribution
- `ggrcst` -- Granger causality test
- `gmccsm` -- Cross-method consistency check (g-formula vs IPW vs g-est)
- `granci` -- Granger causality as MI
- `grc1d` -- Causal (masked) 1D convolution for time-series forecasting
- `gscmcl` -- Generalized Synthetic Control with interactive fixed effects
- `hmc1d` -- Causal 1D convolution: output at time t only depends on t'<=t
- `htgcrf` -- Hetero-causal forest with monotonicity
- `ident` -- Identifiability conditions for causal effects from observational data
- `immid` -- Index of moderated mediation
- `ipsiMed` -- Interventional ψ in causal forests
- `ivcrt` -- Three conditions for a valid instrument Z for causal effect of X on Y
- `kmclm` -- Causal LM next-token cross-entropy loss (GPT-style)
- `kmprf` -- Prefix-LM attention mask: bidirectional over prefix, causal over completion
- `longMd` -- Longitudinal mediation (cross-lagged)
- `mcausm` -- Multi-mediator causal mediation analysis
- `mdian` -- Mediation analysis: decompose total effect into direct and indirect
- `medML` -- Double ML mediation
- `medSEM` -- SEM-based mediation (lavaan style)
- `medfm` -- Pearl's mediation formula
- `medstg` -- Sequential / chain mediation X -> M1 -> M2 -> Y
- `mivbnd` -- Manski monotone instrumental variable bounds
- `mlmMd` -- Multilevel (1-1-1, 2-1-1) mediation
- `msmest` -- Marginal structural model fit by inverse-probability-of-treatment weighting
- `msmiv2` -- MSM with instrumental variables
- `mssm` -- Marginal structural mediation model
- `nchunk` -- Chunked causal attention for long-context efficiency
- `nemed` -- Nested counterfactual mediation effect
- `nonresp` -- Nonresponse adjustment via response propensity
- `npstm` -- Nonparametric TMLE for survival treatment effect
- `plcbo` -- Placebo treatment refutation test for causal estimates
- `potef` -- Individual treatment effect (ITE) using potential outcomes notation
- `prsmtd` -- Sequential propensity-score matching
- `pscme` -- Path-specific causal effect for multiple mediators
- `qbcfgr` -- Quantile-balanced causal forest for distributional treatment effects
- `rgztf` -- Z-transform of a causal discrete-time sequence
- `rhomed` -- Rho critical value where mediation effect -> 0
- `rng032` -- Causal continuous-time convolution form (lower limit 0, upper limit t)
- `rng033` -- Equivalent causal continuous-time convolution with swapped arguments
- `rng036` -- Discrete-time causal convolution sum
- `rng037` -- Equivalent discrete-time causal convolution with swapped arguments
- `rng049` -- Laplace transform of a causal finite-duration h(t) over [0,T]
- `rng053` -- Z-transform of a causal FIR system of length N (transfer function)
- `rng103` -- Running integral of a causal signal over [0, t]
- `rng196` -- Noncausal least-squares second derivative used to detect the dicrotic notch
- `scmaba` -- Synthetic Control Method (Abadie-Diamond-Hainmueller)
- `scmdf` -- Structural causal model (SCM) definition: (U, V, F) triple
- `sensIM` -- Imai-Keele sensitivity to unmeasured confounding
- `seqM` -- Sequential (causally ordered) mediators
- `shrtgr` -- Shrinkage propensity model via Bayesian prior
- `snsmed` -- Sensitivity analysis for unmeasured confounding
- `spwgts` -- Spline-based propensity weights
- `survcfg` -- Causal survival forest from grf
- `survmd` -- Mediation with survival outcome
- `tdmed` -- Two-dimensional mediation effect with M1 and M2
- `tmlivc` -- TMLE for instrumental variable LATE
- `tmllng` -- TMLE for longitudinal data with time-varying treatments and confounders
- `tmlmed` -- TMLE for natural direct + indirect mediation effects
- `tmlpoy` -- Propensity-only TMLE -- robust if Q misspecified
- `tmlqct` -- TMLE for quantile treatment effects
- `tmlsen` -- TMLE bias bound under unmeasured confounding
- `tmltrt` -- TMLE with propensity truncation
- `tmltvc` -- TMLE under time-varying confounding (non-longitudinal sequential)
- `trnfen` -- Transfer entropy (info-theoretic causality)
- `unitnr` -- Unit nonresponse propensity weighting
- `weakid` -- Weak identification check for mediation
- `backDR` -- Back-door adjustment
- `causmrop` -- G-formula (parametric) standardised mean
- `drlnr` -- DR-learner: doubly robust meta-learner for CATE
- `medFront` -- Front-door criterion
- `msmphr` -- Marginal structural Cox model with IPTW
- `pluginM` -- Plug-in g-computation NIE
- `sntmod` -- Sequential targeted parametric g-formula
