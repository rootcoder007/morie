---
title: 'MORIE: A Multi-Domain Scientific Computing Toolkit, with the MRM Framework for Canadian Carceral, Police, and Oversight Data'
tags:
  - Python
  - R
  - causal inference
  - scientific experimentation
  - criminology
  - carceral analysis
  - signal processing
  - cryptography
  - spatial statistics
  - psychometrics
  - MRM
  - Mandela Rules
  - terminal user interface
authors:
  - name: Vansh Singh Ruhela
    orcid: 0009-0004-1750-3592
    affiliation: 1
affiliations:
  - name: Centre for Criminology and Sociolegal Studies, University of Toronto
    index: 1
date: 9 May 2026
bibliography: paper.bib
suppress-bibliography: true
---

# Summary

MORIE (**M**ethods for **O**bservational **I**nference and
**R**obust **A**nalysis of **I**nterventions in **S**cientific
**E**xperimentation) is an open-source, dual-language (Python and R)
scientific-computing toolkit that supports observational
inference and intervention analysis across a wide range of
scientific-experimentation contexts, with sociolegal data
analysis (carceral, police, and oversight) as its named
flagship domain. The package is
home to the MRM (Multilevel Reconciliation Methodology) framework, a
multi-source mathematical foundation that integrates five distinct
Canadian carceral, police, and oversight data streams under one
set of estimators [@ruhela2026dlrm]: provincial Ontario
restrictive-confinement microdata (OTIS), federal Structured
Intervention Unit reports and academic replications by
Sprott–Doob–Iftene [@sprottdoob2021torture;
@sprottdoobiftene2021iedm], the Ontario Special Investigations
Unit (police-oversight) corpus, the Toronto Police Service
open-data crime categories with the Statistics Canada Crime
Severity Index [@wallace2009csi], and federal Corrections and
Conditional Release Statistical Overview tables introduced via
Doob's *T-539-20* affidavit [@doob2020affidavit]. MRM is
implemented in MORIE as a set of MRM modules — a 10-estimator
per-individual causal ensemble paired with a GEE
clustering grid, a Doob $\chi^{2}$ family for aggregate
contingency tables, a Goffmanian institutional-churn analysis
suite [@goffman1961asylums], and a Mandela Rules classifier
[@un2015mandela] that operates at both the federal level (full
out-of-cell-hour operationalization) and the provincial level
(duration-only proxy). Beyond the sociolegal flagship domain, MORIE provides
general-purpose causal-inference estimators (IPW, AIPW,
double machine learning [@chernozhukov2018double],
propensity-score matching, sensitivity analysis), survey-weighted
inference [@lumley2010complex], signal processing and spectral
analysis (with applications to forensic audio and biomedical
signals), homomorphic-deconvolution methods, cryptographic
primitives, spatial statistics (Hawkes self-exciting processes,
Moran's *I*, Ripley's *K*, Getis–Ord $G^{\ast}$), stochastic
physics of crime (reaction-diffusion, Lévy flight, urban
scaling), and classical-test-theory and item-response-theory
psychometrics. The toolkit ships with 41 built-in Canadian
datasets, runs entirely from the terminal via a 10-screen
Textual TUI, and supports a multi-provider LLM chain (local
Ollama, free OllamaFreeAPI, Gemini, OpenAI-compatible) with a
vendored TurboQuant [@zandieh2026turboquant] KV-cache compression
implementation for offline inference.

# Statement of need

Sociolegal data analysis — MRM's named home domain — sits
between criminology, statistics, and law, and is produced by disjoint government agencies under
disjoint legal frameworks. In the Canadian context alone, an
analyst comparing provincial restrictive-confinement (OTIS),
federal Structured Intervention Units (CSC), Ontario police
oversight (Special Investigations Unit), and Toronto Police
Service open data must currently stitch together five separate
data dictionaries, five release cadences, and five formats
(microdata CSV, PDF report, HTML director's report, GeoJSON,
JSON). No primary key joins these sources, and most existing
software targets one of them in isolation. MORIE addresses this
gap by providing a single namespace and a single set of
estimators that span all five sources, with MRM/RF as the
unifying mathematical layer.

The package's reach extends well beyond the sociolegal
flagship into adjacent scientific-experimentation contexts:

- **General observational inference.** Researchers in any field
  needing IPW, AIPW, double-machine learning, propensity-score
  matching, instrumental variables, regression discontinuity, or
  E-value sensitivity analysis [@vanderweele2017evalue] can use
  MORIE without engaging the carceral / oversight modules.
- **Forensics and biomedical signals.** The
  `signal_processing` and `homomorphic_deconvolution` modules
  expose spectral analysis, cepstral methods, wavelets, and the
  blind-deconvolution stack relevant to forensic audio and
  biosignal applications.
- **Cryptography.** A `crypto` module collects classical and
  modern primitives suitable for teaching and for low-stakes
  research workflows.
- **Spatial statistics, statistical physics, and
  psychometrics.** Each is a self-contained module that can be
  used independently.

MORIE is terminal-first by design. Public-sector analysts and
researchers in secure or air-gapped environments often lack a
graphical desktop or stable internet, and most existing causal-
inference and observational-analysis tools assume a browser-based
notebook front end. MORIE's Textual TUI provides interactive
data exploration, pipeline execution, polyglot REPL (Python / R /
Shell with bidirectional variable bridging), and LLM-assisted
analysis without a web browser, and the vendored TurboQuant
KV-cache compression makes local inference practical on
consumer hardware.

# Key features

## MRM framework (the MRM modules)

The carceral, police, and oversight analyses are organised
around the MRM framework. The MRM modules, MORIE's
implementation of MRM, cover:

- An **aggregate IRR family** for Poisson- and
  negative-binomial-distributed counts with a log offset and
  fixed-effects strata.
- A **per-individual 10-estimator causal ensemble** (IPW–Hájek,
  AIPW, IRM-DML with cross-fitting, PSM, PSM with
  subclassification, AIPW with SuperLearner
  [@vanderlaan2007superlearner], PLR-DML, ATE/ATT/ATC) with
  *canonical* and *dual* (naive-arm) sensitivity formulations
  on three OTIS individual-level files.
- A **GEE clustering grid** with priority-ordered selection
  among Poisson/NB families × Exch/Indep working correlations,
  for high-cardinality groupings such as the 25 Ontario
  correctional institutions or the 5 CSC federal regions.
- A **Doob $\chi^{2}$ family** that applies Pearson $\chi^{2}$
  and Cramér's *V* to every meaningful 2-way slice of OTIS
  aggregate tables.
- A **Goffmanian institutional-churn suite** operationalising
  Goffman's [@goffman1961asylums] *total institutions* thesis
  via Gini concentration on repeat placements, Pareto vs
  log-normal AIC for embedding, joint $\chi^{2}$ for
  mortification, and intra-fiscal-year first-order Markov
  transition matrices.
- A **Mandela Rules classifier** that runs at the federal level
  with the full Sprott–Doob operationalization and at the
  provincial level with a transparent duration-only proxy. The
  cross-jurisdiction comparison reveals that Ontario provincial
  Mandela "torture"-classified proportions rise monotonically
  from 12.5 % in fiscal 2023 to 20.6 % in fiscal 2025, exceeding
  the federal SIU rate of 9.9 % that Sprott and Doob found for
  fiscal 2019–2020 [@sprottdoob2021torture].

## Crime-data spatial-temporal stack

A general Hawkes self-exciting process with exponential, gamma,
Weibull, and Lomax (power-law) excitation kernels, sinusoidal or
constant baselines, and a time-rescaling Q–Q diagnostic
[@daley2003pointprocess]. Companion spatial statistics include
Moran's *I* (global, local, bivariate), Ripley's *K*, and
Getis–Ord $G^{\ast}$. A stochastic-physics-of-crime stack adds
Short–Brantingham–D'Orsogna reaction-diffusion
[@shortbrantingham2008], Lévy-flight scaling, urban scaling
[@bettencourt2007], and Lotka–Volterra police–crime dynamics.

## Ontario SIU automation

A polite scraper, parser, normaliser, and writer for the Ontario
Special Investigations Unit director's-report corpus. Output is
a 65-column canonical CSV keyed on the SIU case number
(`YY-XXX-NNN`); seven RichResult-emitting analysers cover
demographics, decision timing, mental-health-and-race
indicators, by-police-service breakdowns, and the case-count
panel.

## General-purpose modules

Causal inference (`causal`, `effects`, `investigation`),
survey-weighted estimation (`survey`), psychometrics (`psymet` —
Cronbach's $\alpha$, McDonald's $\omega$, KMO, Bartlett, parallel analysis,
composite reliability, AVE), signal processing
(`signal_processing`, `homomorphic_deconvolution`), spatial
statistics (`spatial`), genomics utilities (`genomics`),
cryptographic primitives (`crypto`), and a 5,800+ file `fn/`
namespace of individual statistical / scientific functions.

These modules are written natively rather than delegating to a
third-party package. Where a method has a canonical source, the
implementation follows that source and the docstring names the
equation, so a reader can check the code against the page the
author wrote. Goodness-of-fit critical values are transcribed from
[@gibbonsChakraborti2011nonparametric]; the multivariate GARCH
family follows [@bollerslev1990ccc] for constant conditional
correlation and [@engle2002dcc] for the dynamic case; the tree
ensembles follow [@breiman2001randomforests], [@friedman2001greedy]
and [@hastie2009esl], with the regularised split objective of
[@chenGuestrin2016xgboost]; the support-vector solver is the
decomposition of [@changLin2011libsvm] with the working-set
selection of [@fanChenLin2005wss]; mediation sensitivity follows
[@imaiKeeleYamamoto2010mediation], whose partial-$R^{2}$
parameterisation is due to [@imbens2003sensitivity]; the
compositional diagnostics follow [@aitchison1986compositional] and
the spurious-correlation argument of [@pearson1897spurious];
collinearity diagnostics follow [@belsley1980regression]; the
classification metric is [@matthews1975comparison]; and the
point-pattern tests follow [@diggle2003pointpatterns] and
[@schabenberger2005spatial]; and the multivariate portmanteau test
follows [@hosking1980multivariate], whose modified statistic is
restated as equation (9) of [@mahdi2020portes];
and the table-inertia chi-square follows
[@nenadicGreenacre2007ca] and [@greenacre1984correspondence];
the kernel two-sample test follows [@gretton2012kernel]; and the
Wasserstein two-sample test follows [@ramdas2017wasserstein], using the
one-dimensional closed form of [@villani2009optimal];
covariate-balance diagnostics follow [@austin2009balance]; space-time
interaction follows [@jacquez1996knn]; and second-order point-pattern
analysis follows [@ripley1977modelling];
differential item functioning follows [@shealyStout1993sibtest]; and
panel cointegration follows [@pedroni1999critical].

Three implementations depend on numerical tables their authors
published rather than on formulas alone: the Lilliefors and
Anderson-Darling critical values of
[@gibbonsChakraborti2011nonparametric], and Pedroni's adjustment
terms [@pedroni1999critical]. These are transcribed into the source
with their citation, and their limits are enforced rather than
smoothed over: Pedroni's table has no single-regressor row, so a
bivariate panel is refused rather than extrapolated, and the
Anderson-Darling table floors at p = 0.01, so smaller values are
reported as bounded rather than invented.

Writing each method twice, once in Python and once in R, is a
deliberate check rather than duplicated effort. Two defects in this
release were found only because the second implementation existed:
an integer overflow in the R Matthews coefficient at sample sizes
above roughly 800, which Python's arbitrary-precision integers
cannot produce, and a column-major unwinding in R that silently
transposed a window specification the Python read correctly.

## Terminal user interface and LLM integration

A Textual TUI with 10 screens, a multi-language REPL with
bidirectional Python / R / Shell variable bridging, and a
provider chain (local Ollama → vendored OllamaFreeAPI client →
Gemini → OpenAI-compatible → keyword-fallback) for
LLM-assisted analysis. The vendored TurboQuant
[@zandieh2026turboquant] KV-cache compression preserves the
unbiased inner-product property required for downstream causal
inference, and a pure-Python emissions tracker covers 213
countries.

# Mathematics

The MRM framework, including the per-individual 10-estimator
ensemble, the GEE priority ordering, the Doob $\chi^{2}$ family,
the Goffmanian institutional-churn estimators, and the Mandela
Rules classifier, is developed in full in [@ruhela2026dlrm].
That paper also documents an empirical replication of all five
published $\chi^{2}$ statistics from the three Sprott–Doob–Iftene
academic reports, reproducing each value to within 0.01 of the
published number from the transcribed cell counts.

# Estimator validation

Citing a source is not evidence that the code implements it. Every
estimator added during the current re-implementation audit is
checked against a design whose answer is known in closed form,
measured over repeated samples rather than a single seed, and the
measured numbers are recorded here and asserted in the test suite.

| Estimator | Design truth | Estimate | MC s.d. | Mean reported s.e. |
|---|---|---|---|---|
| Survival TMLE, $S_1(t_0) - S_0(t_0)$ | 0.336 | 0.335 | 0.024 | 0.025 |
| Callaway--Sant'Anna ATT (staggered) | 2.05 | 2.05 | -- | -- |
| Two-way fixed effects, same design | 2.05 | 1.55 | -- | -- |
| Variable importance, $R^{2}$, strong signal | 0.667 | 0.675 | 0.030 | -- |
| Variable importance, $R^{2}$, weak signal | 0.167 | 0.174 | 0.026 | -- |
| Variable importance, $R^{2}$, null | 0.000 | 0.003 | 0.018 | -- |
| G-estimation, structural nested mean model | 1.500 | 1.507 | -- | -- |
| Kaplan--Meier median, exponential(10) | 6.93 | 7.14 | -- | -- |
| Kaplan--Meier $S(10)$, same design | 0.3679 | 0.3721 | -- | -- |
| Kozachenko--Leonenko entropy, $d = 1$ | 1.4189 | 1.4166 | -- | -- |
| Kozachenko--Leonenko entropy, $d = 4$ | 5.6758 | 5.6017 | -- | -- |
| Bridge sampling, $\log Z$ | 2.0000 | 2.0000 | -- | -- |
| Mediation, product of coefficients $ab$ | 0.300 | 0.300 | -- | -- |
| BayesC$\pi$, proportion of null markers | 0.950 | 0.925 | -- | -- |
| MINE mutual information, $\rho = 0.6$ | 0.2231 | 0.2100 | -- | -- |
| MINE mutual information, $\rho = 0.9$ | 0.8304 | 0.6811 | -- | -- |
| LD50, probit, known design | 4.000 | 4.001 | 0.0879 | -- |
| Fieller interval coverage, LD50 | 0.950 | 0.953 | -- | -- |
| Median SE, normal electorate, $n=4000$ | 0.01982 | 0.02000 | 0.01971 | -- |
| Dual-frame total, Hartley | 504.4 | 505.2 | 15.0 | -- |
| Same population, naive pooling | 504.4 | 608.8 | 10.7 | -- |
| Bracken abundance, species 1 of 3 | 0.5000 | 0.5016 | -- | -- |
| Bracken abundance, species 2 of 3 | 0.3000 | 0.2991 | -- | -- |
| Bracken abundance, species 3 of 3 | 0.2000 | 0.1994 | -- | -- |
| FACE eigenvalue 1, Karhunen--Loeve | 1.0000 | 1.0033 | -- | -- |
| FACE eigenvalue 2, same design | 0.5000 | 0.5084 | -- | -- |
| FACE noise variance $\sigma^{2}$ | 0.0900 | 0.0931 | -- | -- |
| W-NOMINATE ideal points, 1-D correlation | 1.0000 | 0.9897 | -- | -- |
| W-NOMINATE ideal points, 2-D (Procrustes) | 1.0000 | 0.9835 | -- | -- |
| AR(1) SE inflation, $\rho = 0.8$ | 3.000 | 3.008 | -- | -- |
| AR(1) SE inflation, $\rho = 0.9$ | 4.359 | 4.355 | -- | -- |
| Text-confounded ATE, adjusted | 1.0000 | 1.0541 | 0.0903 | -- |
| Same design, unadjusted | 1.0000 | 2.5381 | -- | -- |
| RESET $F$, response scaled $\times 10^{6}$ | 315.076 | 315.076 | -- | -- |
| Exact $\binom{100}{50}$ vs base R `choose` | 100891344545564193334812497256 | exact (R: ...563076171808112640) | -- | -- |
| $p(1000)$, both languages | 32 digits | agree at all 32 | -- | -- |
| Bell(25), both languages | 4638590332229999353 | agree at all 19 | -- | -- |
| Goodman identity vs enumeration | 0 | 0 residual, 200 colourings | -- | -- |
| $R(3,3)$ by exhausting $K_6$ | 6 | 6 (min 2 mono triangles) | -- | -- |
| AIPW ATE against Hahn's efficiency bound | 1.000 | 0.999 | 0.0559 | 0.0542 |
| Minimax regret constant $\max_t t\Phi(-t)$ | 0.169971 | 0.169971 | -- | -- |
| Plug-in treatment rule, worst-case regret | 0.007601 | 0.007625 | -- | -- |
| Private mean interval coverage, honest | 0.950 | 0.982 | -- | -- |
| Private mean interval coverage, naive | 0.950 | 0.336 | -- | -- |

Several identities are checked as identities rather than
approximately, and hold to machine precision: the mediation relation
$c - c' = ab$ for a continuous least-squares outcome (residual
$0.00 \times 10^{0}$), the restricted-likelihood relation
$\ell_{REML} = \ell_{ML} - \tfrac12\ln|X'V^{-1}X| + \tfrac{p}{2}\ln 2\pi$,
the Krogh--Vedelsby decomposition of ensemble error into mean member
error minus ambiguity ($10^{-10}$), the Goodman--Bacon weights
summing to one with zero residual against the regression coefficient,
and the Aalen--Johansen partition in which the cumulative incidences
over all causes plus the overall survival equal one at every time
($2.7 \times 10^{-14}$).

The MINE rows are included because they show the estimator failing in
the direction it must. It is a lower bound on mutual information, and
it understates progressively as the dependence strengthens -- accurate
at $\rho = 0.6$, short by a fifth at $\rho = 0.9$. That is the
sample-size limit of McAllester and Stratos [-@mcallester2020formal]
in practice, not a defect of the implementation, and an estimator that
did not show it would be the suspicious one.

Three results are worth stating separately because each is a
property an implementation can fail silently.

*The two-way fixed-effects gap is not noise.* On a staggered
adoption design constructed so that the true average effect on the
treated is 2.05, the two-way fixed-effects regression returns 1.55.
The Goodman-Bacon decomposition [@goodmanBacon2021did] implemented
here identifies why: exactly one seventh of the estimation weight
sits on the comparison of later-treated units against
already-treated ones, which returns 0.25. The decomposition is
verified as an identity rather than assumed -- the weights sum to 1
and the residual against the regression coefficient is zero to
machine precision. Callaway and Sant'Anna [-@callawaySantanna2021],
the Borusyak-Jaravel-Spiess imputation estimator
[@borusyak2024revisiting] and Wooldridge's extended two-way
estimator [@wooldridge2021twoway] all recover 2.05, the last two
agreeing with each other to $2 \times 10^{-14}$.

*Sample-splitting is what makes a null importance testable.* On a
design where the tested variable has exactly zero importance,
sample-splitting holds the type-I error at 0.055 against a nominal
0.05 over 200 replications. Omitting it collapses the estimator's
Monte Carlo standard deviation from 0.018 to 0.001 and the test
rejects 0 times out of 200 -- the degeneracy Williamson and
colleagues [-@williamson2023general] warn about, reproduced as a
test assertion rather than paraphrased in a comment.

*Influence functions are checked numerically, not by eye.* Each
predictiveness gradient is verified against an exact Gateaux
derivative, obtained by appending a duplicate observation so the
tilt $\epsilon = 1/(n+1)$ carries no sampling error. The
discrepancy is then the $O(\epsilon)$ curvature term and falls with
the sample size as it must: 0.065, 0.0080 and 0.0021 at $n$ = 250,
1000 and 4000. This check is also what established that the
gradient printed in the source's own appendix is stated up to an
additive constant, so the mean-zero form is used instead --
matching what the authors' reference implementation computes.

*A bound is only useful if something is measured against it.* The
semiparametric efficiency bound for the average treatment effect
[@hahn1998propensity] and the local asymptotic minimax regret bound
for treatment choice [@hirano2009asymptotics] are both computed here
and then tested for attainment rather than quoted. Over 600
replications at $n = 1500$, the augmented inverse-probability
estimator has a sampling variance 1.064 times the bound while Hajek
weighting sits above 1.3 times it, which is the ordering the theory
requires. The minimax constant is solved from its own stationarity
condition $\Phi(-t) = t\phi(t)$ rather than taken from the
literature's two significant figures, giving 0.169971; simulating
the local experiment the bound describes, the plug-in rule's
worst-case regret comes to 0.007625 against a bound of 0.007601, so
it attains the bound to within 0.3 per cent. The bound is also
decomposed into the term driven by overlap and the term driven by
effect heterogeneity, and the second of these does not fall as
nuisance estimation improves -- reporting an average over a
heterogeneous population has an irreducible price.

*The price of a privacy guarantee has to appear in the interval, and
usually does not.* For a differentially private mean the mechanism
noise is a second source of uncertainty on top of sampling error. An
interval that accounts only for the sampling error is the one
commonly reported. At $\epsilon = 0.05$ on $n = 200$ observations,
that naive interval covers the true mean 33.6 per cent of the time
against a nominal 95; the interval that carries the mechanism noise
covers 98.2 per cent. The same module also corrects a claim this
work initially made in the other direction: the clipping width is
usually described as trading bias against noise, and it does not
always do so. Clipping a symmetric distribution with a window centred
on its mean removes equal mass from both tails, so the bias cancels
at every width -- measured on a standard normal, it stays below 0.006
while the noise grows four-hundred-fold. The trade-off requires skew,
and the reported error curve now flags the case where no interior
optimum exists rather than returning the edge of the search grid as
though it were one.

*A diagnostic that fires on noise is worse than none.* The
goodness-of-fit check on a quantal dose-response fit is conventionally
stated as a heterogeneity *factor* -- residual deviance over its
degrees of freedom -- with values above 1 taken as evidence that
subjects did not respond independently. That rule is not a test. The
deviance has expectation equal to its degrees of freedom, so over 400
correctly specified replications the factor averaged 0.69 and still
exceeded 1 in 20.5 per cent of them. Keying the warning to the
:math:`\chi^2` tail probability instead brings the false-alarm rate to
where it belongs.

Doing so exposes a second effect, which is a property of the assay
design rather than of the code. A dose group whose fitted probability
sits within $10^{-5}$ of 0 or 1 contributes essentially no deviance
while still spending a degree of freedom, so the reference
distribution is wrong in a predictable direction. Holding everything
else fixed and varying only the dose range:

| Dose range | Fitted probabilities | Mean null $p$ | Fires at 5% |
|---|---|---|---|
| 0.5 to 32 (saturated ends) | 0.00002 to 0.99998 | 0.645 | 1.5% |
| 2 to 8 (all informative) | 0.083 to 0.917 | 0.479 | 5.5% |

The narrow design is essentially uniform, as a correctly calibrated
test should be. The wide one is conservative: it under-warns rather
than over-warns, which is the safe direction, but it is not the
nominal level and the implementation says so rather than quietly
reporting a $p$-value it cannot support.

# Numerical failures that do not announce themselves

Three defects found during this work share a shape: the code runs,
returns a plausible number, and is wrong. None would have been caught
by inspection, and two were in estimators that had passed every
correctness test written for them. They are recorded because the
failure mode is more transferable than the fixes.

*A non-concave objective makes the starting point part of the method.*
The spatial voting model is fitted by alternating between legislators
and roll calls. Each half-step is a probit and cannot decrease the
likelihood, which makes the procedure look safe; the joint objective
is not concave, so it is not. Fitting a simulated 120-member chamber
with known ideal points:

| Start | Sweeps | Log-likelihood | Correlation with truth |
|---|---|---|---|
| Random | 60 | $-19{,}801$ | 0.47 |
| Random | 200 | $-7{,}045$ | 0.965 |
| Leading singular vectors | 26 | $-6{,}913$ | 0.990 |

The first row is the dangerous one. It reports an ideal-point
configuration that is superficially reasonable and is in fact almost
uninformative, and only the convergence flag distinguishes it. Seeding
from the leading singular vectors of the centred vote matrix -- which
alone correlate 0.92 with the truth before any fitting -- reaches a
better optimum in a fraction of the sweeps. The lesson taken into the
implementation is that a stopped iteration must say so loudly and
report the per-sweep change in the objective, not merely a boolean.

*An imputation choice can dominate the estimator it feeds.* The
functional covariance smoother must hold out the diagonal, which
carries the measurement-error variance. Holding it out requires
filling it with something, and the obvious first choice -- the mean of
the rest of the row -- is badly wrong, because the row mean of
$C(s,t)$ over $s$ is nowhere near $C(t,t)$. On a two-component
Karhunen--Loève design with $\sigma^2 = 0.09$ it returned 0.334, and
left 13 per cent of the eigenvalue mass negative on an operator that
cannot have any. Imputing the diagonal from the smoothed surface and
iterating to a fixed point gives 0.093, with negative mass at
$6 \times 10^{-4}$ of the total. The eigenvalues and eigenfunctions
were accurate under *both* versions -- 0.973 and 0.487 against 1.0 and
0.5 -- so every test aimed at the headline quantities passed while the
noise estimate was off by a factor of nearly four.

*Conditioning fails quietly and scales with the data.* The RESET test
augments a design with powers of the fitted values. Cubing an unscaled
$\hat y$ makes the condition number of the augmented cross-product
grow with the **sixth** power of the response scale. On a quadratic
design whose correct statistic is $F = 315.08$, multiplying the
response by 100 returned 229.06, by 1000 returned 125.59, and on a
second design the statistic came out *negative*. Since $F$ is
invariant to the scale in exact arithmetic, every one of those numbers
is a pure artefact of floating point. Normalising $\hat y$ before
taking powers does not change the column span, and solving by QR
rather than through the normal equations avoids squaring the condition
number again; together they hold the statistic at 315.075993 across
six orders of magnitude of response scale. This one had been shipping
in the R implementation, and was found only because the Python port
disagreed with it.

# Exactness

Two languages, two different failure modes, and neither announces
itself.

*R has no exact integer beyond $2^{53}$.* This is usually described as
a precision limit, which understates it. `2^53 + 1 == 2^53` evaluates
to `TRUE`, and base R's own `choose(100, 50)` returns
100891344545563076171808112640 where the exact value is
100891344545564193334812497256 --- wrong from the thirteenth
significant digit, off by more than $10^{15}$, with nothing in the
output to say so. Every exact count in the combinatorics modules is
therefore routed through an arbitrary-precision integer layer written
from scratch for this package: sign plus a little-endian base-$10^6$
limb vector, with schoolbook multiplication carrying row by row so
every intermediate stays below $10^{12} + 10^6$ and remains an exactly
representable double. Binomial coefficients use the multiplicative
recurrence $\prod_{i=1}^{k} (n-k+i)/i$, in which each partial product
is exactly divisible by $i$, so the computation needs only
multiplication and division by a small integer and never long
division. `gmp` is deliberately not a dependency.

The parity tests compare **decimal strings**, not doubles. That choice
is the point: comparing through a double would let a silent loss of
low-order digits pass, which is precisely the failure being guarded
against. On that basis $p(1000)$ agrees across the two languages at all
32 digits, and Bell(25) $= 4638590332229999353$ at all 19.

*Two subtleties surfaced while building it, both corrections to our own
first version.*

The rule $|v| \le 2^{53}$ for "exactly representable as a double" is
sufficient but **not necessary**, and treating it as necessary
mislabels exact values as lossy. $20! \approx 2.4 \times 10^{18}$ sits
far above the threshold and is nonetheless exact, because it is
$2^{18}$ times an odd number and its low-order bits are already zero.
The test now round-trips the value through a double; the threshold rule
is retained separately with its status stated.

The second is a cross-language trap rather than a numerical one. In R
the integer-division operator `%/%` binds *more tightly* than `*`, so
`k * (3*k - 1) %/% 2` parses as `k * ((3*k - 1) %/% 2)` and evaluates
to 4 at $k = 2$, where the second pentagonal number is 5. Python's `//`
shares precedence with `*` and associates left to right, so the
character-for-character identical expression is correct there. The
consequence in the partition-counting routine was $p(4) = 4$ instead of
5 and every subsequent value wrong. It was caught by the cross-language
parity test and by nothing else: the R code is well-formed, runs
without warning, and returns plausible-looking integers.

*A third case shows the same layer being used to check a theorem
rather than to report one.* The hook length formula states that
$f^\lambda = n! / \prod_{(i,j)} h(i,j)$ counts standard Young tableaux
of shape $\lambda$. That a product of hook lengths should divide $n!$
exactly is not obvious, and it is the kind of claim an implementation
can appear to satisfy while doing something else. Both languages
therefore compute the count from *prime exponents* --- Legendre's
formula for the exponent of $p$ in $n!$, minus the exponents obtained
by factoring each hook --- which cannot overflow at any shape, and
then multiply the result back against the hook product and compare it
with $n!$ in arbitrary precision. The residual is reported rather than
assumed to be zero. At shape $(10, 9, \ldots, 1)$, 55 cells, the two
languages agree on all 35 digits of
$f^\lambda = 44261486084874072183645699204710400$, on all 39 digits of
the hook product, and on all 74 digits of $55!$; the same holds at the
$8 \times 8$ square, 64 cells. None of those three numbers is
representable as a double, and none of the comparisons would mean
anything if they were made numerically.

The corollary $\sum_\lambda (f^\lambda)^2 = n!$ is then checked as an
identity in both languages, which tests the tableau count against
Robinson--Schensted rather than against a table of remembered values.
Robinson--Schensted itself is a bijection, so it is round-tripped over
*every* permutation of $n \le 6$ in both languages --- 873 words ---
and the recovered permutation must equal the original exactly. A
formula can be checked only against another formula; a bijection can
be checked against itself.

The same standard applies to counting up to symmetry. Burnside's lemma
is frequently taught as "divide by the symmetry", which is wrong
whenever some arrangements have symmetry of their own: two-colour
necklaces of length four admit 16 colourings and 4 rotations but 6
orbits, not 4. Both implementations return that naive quotient
alongside the correct count and flag the disagreement, and both are
checked against orbits enumerated *directly* --- every colouring
generated, every image under every group element marked --- for all
cyclic groups to $n = 6$ at two and three colours. The closed form for
necklaces, $n^{-1} \sum_{d \mid n} \varphi(n/d) k^{d}$, is a shortcut
over that enumeration, and shortcuts are where errors hide, so it is
computed both ways and the two are compared at every $n \le 12$ for
$k \in \{2, 3, 4\}$.

One more R-specific trap surfaced here, and it is the mirror image of
the `%/%` one. The idiom `for (p in 2:floor(sqrt(n)))` for a sieve is
correct for every $n \ge 4$ and silently reverses for $n < 4$, where
`2:1` counts *down* and hands `seq.int` a negative step. In Python
`range(2, int(n**0.5) + 1)` is simply empty. The failure was loud when
it came --- `wrong sign in 'by' argument` --- but it appeared only in
the three tests that reached a partition of $n \le 3$, and only
because the suite enumerated every partition of every $n$ from 1
rather than starting where the interesting shapes are.

The analytic-combinatorics modules extend the discipline from counts
to *asymptotics*. An asymptotic estimate is a theorem about exact
coefficients, so every estimate is computed alongside the exact value
it approximates: Hardy--Ramanujan against $p(n)$ from the pentagonal
recurrence in arbitrary precision (the estimate is still 4.6 per cent
high at $n = 100$, which the fame of the formula tends to obscure);
the transfer theorem's $n^{\alpha-1}/\Gamma(\alpha)$ against the exact
product $\prod (\alpha + i - 1)/i$; Stirling's series against
$\ln\Gamma(n+1)$ with the alternating-envelope bound *checked*, not
cited --- which surfaced its own subtlety, that the bound outruns
double precision by $n = 50$ and a naive check would measure the
rounding of the comparison rather than the series. The rounding
identity $D_n = \mathrm{round}(n!/e)$ is verified by *two independent
exact routes*: Python encloses $e$ in rational arithmetic and bounds
the distance directly, while R recomputes $D_n$ from the
inclusion--exclusion sum and requires digit-for-digit agreement with
the recurrence. Two derivations of the same claim are a stronger check
than one derivation copied twice.

That is the general argument for the parity discipline. A second
implementation in a second language is not duplicated effort; it is the
only check that catches defects which are invisible from inside one
language's semantics.

Cross-language agreement is treated as a second, independent
implementation rather than a formality. The targeting step of the
average-treatment-effect estimator is deterministic, so the Python
and R versions are pinned to each other at ten significant digits
on a shared fixture generated by exact integer arithmetic. The
causal forest cannot be pinned that way and is not pretended to be:
R draws subsamples with the Mersenne Twister and the Python forest
with a permuted congruential generator, so the two see different
subsamples by construction, and the parity test asserts that they
recover the same estimand rather than the same bits.

# Acknowledgements

The MRM methodology lineage acknowledges
the federal context provided by Anthony N. Doob's affidavit
[@doob2020affidavit] and the four Sprott–Doob–Iftene
independent academic reports [@sprottdoob2020operation;
@sprottdoob2020covid; @sprottdoob2021torture;
@sprottdoobiftene2021iedm].

The general-purpose modules implement published methods, and the
authors of those methods are the reason the implementations can be
checked rather than merely trusted. The signal-processing and
waveform-complexity modules follow Rangayyan and Krishnan
[-@rangayyan2024biomedical]; the fractal and complexity estimators
follow Higuchi [-@higuchi1988approach], Grassberger and Procaccia
[-@grassberger1983strangeness], and Peng and colleagues
[-@peng1994mosaic]; the entropy estimators follow Pincus
[-@pincus1991approximate] and Richman and Moorman
[-@richman2000physiological]; the largest Lyapunov exponent follows
Rosenstein, Collins and De Luca [-@rosenstein1993practical]; the
spectral estimators follow Welch [-@welch1967use] and, for the
autoregressive route, Burg [-@burg1975maximum] and Marple
[-@marple1987digital]; the adaptive noise canceller follows Widrow
and Stearns [-@widrow1985adaptive]; the wavelet transforms follow
Percival and Walden [-@percival2000wavelet] and the shrinkage
thresholds Donoho and Johnstone [-@donoho1994ideal]; the heart-rate
variability measures follow the Task Force of the European Society
of Cardiology and the North American Society of Pacing and
Electrophysiology [-@taskforce1996heart]; the polytomous
item-response model follows Samejima [-@samejima1969estimation]; the
genomic relationship matrices follow VanRaden
[-@vanraden2008efficient]; the global spatial autocorrelation
statistic follows Moran [-@moran1950notes] in the formulation of
Schabenberger and Gotway [-@schabenberger2005spatial]; the
correlation variance-stabilising transform follows Fisher
[-@fisher1921probable]; the multivariate information measures follow
Watanabe [-@watanabe1960information] and Han [-@han1978nonnegative];
the classical test-theory item statistics follow Nunnally and
Bernstein [-@nunnally1994psychometric]; the variance inflation for
dependent effect sizes follows Hedges, Tipton and Johnson
[-@hedges2010robust]; the family-based association test follows
Spielman, McGinnis and Ewens [-@spielman1993transmission]; the Markov
chain Monte Carlo effective sample size follows Geyer
[-@geyer1992practical]; and the functional data correlation follows
Ramsay and Silverman [-@ramsay2005functional].
The econometric and nonparametric modules follow Horowitz
[-@horowitz2009semiparametric] throughout: the single-index
estimators follow Ichimura [-@ichimura1993semiparametric], Klein and
Spady [-@kleinSpady1993efficient], Powell, Stock and Stoker
[-@powellStockStoker1989semiparametric] and Newey and Stoker
[-@neweyStoker1993efficiency]; the rank estimators follow Han
[-@han1987nonparametric], Sherman [-@sherman1993limiting] and
Cavanagh and Sherman [-@cavanagh1998rank]; the maximum-score
estimators and their choice-based, panel and ordered-response
extensions follow Manski [-@manski1985semiparametric; -@manski1987semiparametric], Horowitz [-@horowitz1992smoothed] and
Kooreman and Melenberg [-@kooremanMelenberg1989maximum], and
Melenberg and van Soest [-@melenbergVanSoest1996parametric]; the
heteroskedastic binary-response estimator follows Lewbel
[-@lewbel2000semiparametric] in the simplified form of Dong and
Lewbel [-@dongLewbel2015simple]; the deconvolution rates follow Fan
[-@fan1991optimal] and the panel deconvolution Horowitz and Markatou
[-@horowitzMarkatou1996semiparametric]; the transformation-model
estimators follow Horowitz [-@horowitz1996distribution] and Chen
[-@chen2002rank], with the baseline hazard built on Breslow
[-@breslow1974covariance]. The empirical-process and semiparametric
inference modules follow Kosorok [-@kosorok2008introduction], the
nonparametric tests Gibbons and Chakraborti
[-@gibbonsChakraborti2011nonparametric], the copula modules Czado
[-@czado2019analyzing], and the volatility and cointegration modules
Tsay [-@tsay2010analysis]. The kernel distribution-function shelf
follows Fauzi and Maesono [-@fauzi2023kernel], with the
distribution-function bandwidth rate due to Azzalini
[-@azzalini1981note]; the density-estimation window widths follow
Silverman [-@silverman1986density] and the Parzen estimator Parzen
[-@parzen1962estimation]; importance sampling follows MacKay
[-@mackay2003information]; the bootstrap follows Efron
[-@efron1979bootstrap], the .632+ correction Efron and Tibshirani
[-@efron1997improvements], and bagging Breiman
[-@breiman1996bagging]. The instrumental-variables modules follow
Wooldridge [-@wooldridge2010econometric] for two-stage least
squares, Anderson and Rubin [-@anderson1949estimation] and Fuller
[-@fuller1977some] for limited-information maximum likelihood, and
Sargan [-@sargan1958estimation] for the overidentification test;
the local average treatment effect follows Imbens and Angrist
[-@imbens1994late] and Angrist, Imbens and Rubin
[-@angrist1996identification]; the doubly robust estimator follows
Robins, Rotnitzky and Zhao [-@robins1994estimation]; double machine
learning follows Chernozhukov and colleagues
[-@chernozhukov2018double]; the semiparametric efficiency bound
follows Hahn [-@hahn1998propensity] and the minimax bound for
treatment choice Hirano and Porter [-@hirano2009asymptotics]; the
differentially private mean follows Dwork and Roth
[-@dwork2014algorithmic] with the finite-sample intervals of Karwa
and Vadhan [-@karwa2018finite]; the permutation language-model
objective follows Yang and colleagues [-@yang2019xlnet]; the
enumerative combinatorics follow Stanley [-@stanley2011enumerative] and
Andrews [-@andrews1984partitions], the monochromatic-triangle identity
Goodman [-@goodman1959acquaintances], and the Ramsey values and bounds
Radziszowski's dynamic survey [-@radziszowski2024ramsey]; the median
voter theorem follows Black [-@black1948rationale]; dual-frame
estimation follows Hartley [-@hartley1962multiple] and Lohr and Rao
[-@lohrRao2000dualframe]; quantal dose-response analysis follows
Finney [-@finney1971probit] with the ratio intervals of Fieller
[-@fieller1954interval]; functional-form testing follows
Ramsey [-@ramsey1969reset]; taxonomic abundance re-estimation follows
Lu and colleagues [-@lu2017bracken]; spatial voting follows Poole and
Rosenthal [-@poole1985spatial]; fast functional covariance estimation
follows Xiao and colleagues [-@xiao2016face]; text-adjusted treatment
effects follow Veitch, Sridhar and Blei [-@veitch2020text]; the
autocorrelation-aware variance follows Geyer [-@geyer1992practical];
and the interaction-weighted event study follows Sun and Abraham
[-@sun2021estimating]. The
partial-identification modules follow Manski
[-@manski1990nonparametric] and Manski and Tamer
[-@manskiTamer2002inference] for worst-case bounds, Imbens and
Manski [-@imbensManski2004confidence] and Stoye [-@stoye2009more]
for confidence intervals on partially identified parameters,
Chernozhukov, Hong and Tamer [-@chernozhukovHongTamer2007] for
criterion-function confidence regions under moment inequalities,
Mogstad, Santos and Torgovitsky [-@mogstadSantosTorgovitsky2018]
for sharp linear-programming bounds, and Lavine [-@lavine1992polya]
for the Polya tree posterior. The robust-statistics modules
follow Huber [-@huber1973robust] for M-regression, Rousseeuw and
Yohai [-@rousseeuwYohai1984robust] for S-estimators, Yohai
[-@yohai1987high] for MM-estimators, Yohai and Zamar
[-@yohaiZamar1988high] for tau-estimators, Rousseeuw and Croux
[-@rousseeuwCroux1993alternatives] for the Qn and Sn scales, and
Theil [-@theil1950rank] and Sen [-@sen1968estimates] for the
median-of-slopes line and its distribution-free interval. The
extreme-value modules follow Hill [-@hill1975simple], Pickands
[-@pickands1975statistical] and Dekkers, Einmahl and de Haan
[-@dekkers1989moment] for the extreme-value index, Hosking
[-@hosking1990lmoments], Hosking, Wallis and Wood
[-@hoskingWallisWood1985estimation] and Hosking and Wallis
[-@hoskingWallis1987parameter] for the L-moment and
probability-weighted-moment fits, Smith and Weissman
[-@smithWeissman1994estimating], Ferro and Segers
[-@ferroSegers2003inference] and Northrop [-@northrop2015efficient]
for the extremal index, and Naveau, Guillou, Cooley and Diebolt
[-@naveau2009modelling] for the madogram estimate of extremal
dependence. The resampling modules follow Efron [-@efron1979bootstrap]
and Efron and Tibshirani [-@efronTibshirani1993introduction] for the
bootstrap, Quenouille [-@quenouille1949approximate] for the
jackknife, and Davison and Hinkley [-@davisonHinkley1997bootstrap]
for the ratio interval; the range-volatility modules follow
Parkinson [-@parkinson1980extreme] and Garman and Klass
[-@garmanKlass1980estimation], the realized-volatility aggregation
Andersen, Bollerslev, Diebold and Labys
[-@andersenBollerslevDieboldLabys2003], and the microstructure-noise
decomposition Zhang, Mykland and A\"it-Sahalia
[-@zhangMyklandAitSahalia2005tale] and A\"it-Sahalia, Mykland and
Zhang [-@aitSahaliaMyklandZhang2005often]. The reliability
coefficients follow Shrout and Fleiss [-@shrout1979intraclass] --
whose Table 2 data is the parity fixture, so the three cases are
checked against the paper's own published values -- with the
variance-component reading of McGraw and Wong
[-@mcgraw1996forming]; the ability estimators follow Lord
[-@lord1980applications] and Samejima [-@samejima1973reliability]
for maximum likelihood and its multimodality under guessing,
Mislevy [-@mislevy1986bayes] and Bock and Aitkin
[-@bock1981marginal] for the Bayes modal estimate, Bock and
Mislevy [-@bock1982adaptive] for expected a posteriori
quadrature, and Warm [-@warm1989weighted] for the weighted
likelihood; and the meta-analysis modules follow Paule and Mandel
[-@paule1982consensus] and Veroniki et al. [-@veroniki2016methods]
for the between-study variance, Viechtbauer
[-@viechtbauer2005bias; -@viechtbauer2010conducting] for the
restricted-likelihood estimator, Hedges, Tipton and Johnson
[-@hedges2010robust] with the small-sample correction of Tipton
[-@tipton2015small] for dependent effect sizes, and Viechtbauer
and Cheung [-@viechtbauer2010outlier] for leave-one-out
influence. The staggered difference-in-differences shelf is
organised around one finding: Goodman-Bacon
[-@goodmanBacon2021did] shows that the two-way fixed-effects
coefficient is a weighted average of every 2x2 comparison in the
panel, including comparisons using already-treated units as
controls, so the decomposition is implemented as an identity and
verified rather than asserted. The replacements follow Callaway
and Sant'Anna [-@callawaySantanna2021] for group-time effects
with clean controls, Sant'Anna and Zhao [-@santannaZhao2020drdid]
and Abadie [-@abadie2005semiparametric] for the doubly-robust
moment, Borusyak, Jaravel and Spiess [-@borusyak2024revisiting]
for the imputation estimator and Wooldridge
[-@wooldridge2021twoway] for the saturated regression that equals
it, Abadie and Gardeazabal [-@abadie2003economic] and Abadie,
Diamond and Hainmueller [-@abadie2010synthetic] for synthetic
control with permutation inference, and Arkhangelsky et al.
[-@arkhangelsky2021synthetic] for synthetic
difference-in-differences; standard errors cluster on the unit
following Bertrand, Duflo and Mullainathan [-@bertrand2004howmuch],
and the ATT-to-ATE step reports the assumption it needs after
S\l{}oczy\'nski [-@sloczynski2022interpreting]. The targeted-learning
and heterogeneous-effect modules follow van der Laan and Rubin
[-@vanderLaanRubin2006tmle] and Gruber and van der Laan
[-@gruberVanDerLaan2010] for the targeting step, Athey and Imbens
[-@atheyImbens2016recursive] and Wager and Athey
[-@wagerAthey2018] for honest causal forests and their
infinitesimal-jackknife variance -- with the finite-tree correction
of Wager, Hastie and Efron [-@wagerHastieEfron2014], without which
the intervals are systematically too wide -- K\"unzel, Sekhon,
Bickel and Yu [-@kunzel2019metalearners] and Nie and Wager
[-@nie2021quasi] for the S-, T-, X- and R-learners, and
Chernozhukov, Demirer, Duflo and Fern\'andez-Val
[-@chernozhukov2018generic] for the group-average effects, whose
warning about sorting on the fitted effect is implemented as a
returned flag rather than left in the prose. Right-censored
outcomes are handled on the discrete hazard scale following Cai and
van der Laan [-@caiVanDerLaan2020], whose clever covariate and
logistic fluctuation are implemented as printed, with the efficient
influence curve of Hubbard, van der Laan and Robins
[-@hubbard2000nonparametric] and the covariate-adjustment argument
of Moore and van der Laan [-@mooreVanDerLaan2009censored].
Algorithm-agnostic variable importance follows Williamson, Gilbert,
Simon and Carone [-@williamson2023general] and the R-squared
special case of Williamson, Gilbert, Carone and Simon
[-@williamson2021nonparametric], whose distinction between
cross-fitting and sample-splitting is enforced rather than
described: without the split the estimator is degenerate under the
zero-importance null, so the returned interval says whether it is
entitled to cover that null.

Each function is implemented against the primary source and its
docstring cites the specific chapter, section or equation, so a
reader can check the implementation against the same page the
author wrote. Where an implementation departs from a printed
statement it says so and shows the measurement that settled it --
for example equation (4.43) of Horowitz
[-@horowitz2009semiparametric] prints ``maximize'' over a sum of
absolute deviations while describing the estimator as a median
regression, and the module minimises, with the discrepancy recorded
in the docstring and asserted in the test suite.

All implementation, all framework design, and all empirical
findings are the work of the framework author.

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-abadie2005semiparametric" class="csl-entry">

Abadie, Alberto. 2005. “Semiparametric Difference-in-Differences
Estimators.” *Review of Economic Studies* 72 (1): 1–19.
<https://doi.org/10.1111/0034-6527.00321>.

</div>

<div id="ref-abadie2010synthetic" class="csl-entry">

Abadie, Alberto, Alexis Diamond, and Jens Hainmueller. 2010. “Synthetic
Control Methods for Comparative Case Studies: Estimating the Effect of
<span class="nocase">California’s</span> Tobacco Control Program.”
*Journal of the American Statistical Association* 105 (490): 493–505.
<https://doi.org/10.1198/jasa.2009.ap08746>.

</div>

<div id="ref-abadie2003economic" class="csl-entry">

Abadie, Alberto, and Javier Gardeazabal. 2003. “The Economic Costs of
Conflict: A Case Study of the Basque Country.” *American Economic
Review* 93 (1): 113–32. <https://doi.org/10.1257/000282803321455188>.

</div>

<div id="ref-aitchison1986compositional" class="csl-entry">

Aitchison, John. 1986. *The Statistical Analysis of Compositional Data*.
Monographs on Statistics and Applied Probability. Chapman & Hall.

</div>

<div id="ref-aitSahaliaMyklandZhang2005often" class="csl-entry">

Aït-Sahalia, Yacine, Per A. Mykland, and Lan Zhang. 2005. “How Often to
Sample a Continuous-Time Process in the Presence of Market
Microstructure Noise.” *Review of Financial Studies* 18 (2): 351–416.
<https://doi.org/10.1093/rfs/hhi016>.

</div>

<div id="ref-andersenBollerslevDieboldLabys2003" class="csl-entry">

Andersen, Torben G., Tim Bollerslev, Francis X. Diebold, and Paul Labys.
2003. “Modeling and Forecasting Realized Volatility.” *Econometrica* 71
(2): 579–625. <https://doi.org/10.1111/1468-0262.00418>.

</div>

<div id="ref-anderson1949estimation" class="csl-entry">

Anderson, Theodore W., and Herman Rubin. 1949. “Estimation of the
Parameters of a Single Equation in a Complete System of Stochastic
Equations.” *The Annals of Mathematical Statistics* 20 (1): 46–63.
<https://doi.org/10.1214/aoms/1177730090>.

</div>

<div id="ref-andrews1984partitions" class="csl-entry">

Andrews, George E. 1984. *The Theory of Partitions*. Cambridge
Mathematical Library. Cambridge University Press.
<https://doi.org/10.1017/CBO9780511608650>.

</div>

<div id="ref-angrist1996identification" class="csl-entry">

Angrist, Joshua D., Guido W. Imbens, and Donald B. Rubin. 1996.
“Identification of Causal Effects Using Instrumental Variables.”
*Journal of the American Statistical Association* 91 (434): 444–55.
<https://doi.org/10.1080/01621459.1996.10476902>.

</div>

<div id="ref-arkhangelsky2021synthetic" class="csl-entry">

Arkhangelsky, Dmitry, Susan Athey, David A. Hirshberg, Guido W. Imbens,
and Stefan Wager. 2021. “Synthetic Difference-in-Differences.” *American
Economic Review* 111 (12): 4088–118.
<https://doi.org/10.1257/aer.20190159>.

</div>

<div id="ref-atheyImbens2016recursive" class="csl-entry">

Athey, Susan, and Guido Imbens. 2016. “Recursive Partitioning for
Heterogeneous Causal Effects.” *Proceedings of the National Academy of
Sciences* 113 (27): 7353–60. <https://doi.org/10.1073/pnas.1510489113>.

</div>

<div id="ref-austin2009balance" class="csl-entry">

Austin, Peter C. 2009. “Balance Diagnostics for Comparing the
Distribution of Baseline Covariates Between Treatment Groups in
Propensity-Score Matched Samples.” *Statistics in Medicine* 28 (25):
3083–107. <https://doi.org/10.1002/sim.3697>.

</div>

<div id="ref-azzalini1981note" class="csl-entry">

Azzalini, Adelchi. 1981. “A Note on the Estimation of a Distribution
Function and Quantiles by a Kernel Method.” *Biometrika* 68 (1): 326–28.
<https://doi.org/10.1093/biomet/68.1.326>.

</div>

<div id="ref-belsley1980regression" class="csl-entry">

Belsley, David A., Edwin Kuh, and Roy E. Welsch. 1980. *Regression
Diagnostics: Identifying Influential Data and Sources of Collinearity*.
John Wiley & Sons. <https://doi.org/10.1002/0471725153>.

</div>

<div id="ref-bertrand2004howmuch" class="csl-entry">

Bertrand, Marianne, Esther Duflo, and Sendhil Mullainathan. 2004. “How
Much Should We Trust Differences-in-Differences Estimates?” *Quarterly
Journal of Economics* 119 (1): 249–75.
<https://doi.org/10.1162/003355304772839588>.

</div>

<div id="ref-bettencourt2007" class="csl-entry">

Bettencourt, Luı́s M. A., José Lobo, Dirk Helbing, Christian Kühnert, and
Geoffrey B. West. 2007. “Growth, Innovation, Scaling, and the Pace of
Life in Cities.” *Proceedings of the National Academy of Sciences* 104
(17): 7301–6.

</div>

<div id="ref-black1948rationale" class="csl-entry">

Black, Duncan. 1948. “On the Rationale of Group Decision-Making.”
*Journal of Political Economy* 56 (1): 23–34.
<https://doi.org/10.1086/256633>.

</div>

<div id="ref-bock1981marginal" class="csl-entry">

Bock, R. Darrell, and Murray Aitkin. 1981. “Marginal Maximum Likelihood
Estimation of Item Parameters: Application of an EM Algorithm.”
*Psychometrika* 46 (4): 443–59. <https://doi.org/10.1007/BF02293801>.

</div>

<div id="ref-bock1982adaptive" class="csl-entry">

Bock, R. Darrell, and Robert J. Mislevy. 1982. “Adaptive EAP Estimation
of Ability in a Microcomputer Environment.” *Applied Psychological
Measurement* 6 (4): 431–44.
<https://doi.org/10.1177/014662168200600405>.

</div>

<div id="ref-bollerslev1990ccc" class="csl-entry">

Bollerslev, Tim. 1990. “Modelling the Coherence in Short-Run Nominal
Exchange Rates: A Multivariate Generalized ARCH Model.” *The Review of
Economics and Statistics* 72 (3): 498–505.
<https://doi.org/10.2307/2109358>.

</div>

<div id="ref-borusyak2024revisiting" class="csl-entry">

Borusyak, Kirill, Xavier Jaravel, and Jann Spiess. 2024. “Revisiting
Event-Study Designs: Robust and Efficient Estimation.” *Review of
Economic Studies* 91 (6): 3253–85.
<https://doi.org/10.1093/restud/rdae007>.

</div>

<div id="ref-breiman1996bagging" class="csl-entry">

Breiman, Leo. 1996. “Bagging Predictors.” *Machine Learning* 24 (2):
123–40. <https://doi.org/10.1007/BF00058655>.

</div>

<div id="ref-breiman2001randomforests" class="csl-entry">

Breiman, Leo. 2001. “Random Forests.” *Machine Learning* 45 (1): 5–32.
<https://doi.org/10.1023/A:1010933404324>.

</div>

<div id="ref-breslow1974covariance" class="csl-entry">

Breslow, Norman E. 1974. “Covariance Analysis of Censored Survival
Data.” *Biometrics* 30 (1): 89–99. <https://doi.org/10.2307/2529620>.

</div>

<div id="ref-burg1975maximum" class="csl-entry">

Burg, John Parker. 1975. “Maximum Entropy Spectral Analysis.” PhD
thesis, Stanford University.

</div>

<div id="ref-caiVanDerLaan2020" class="csl-entry">

Cai, Weixin, and Mark J. van der Laan. 2020. “One-Step Targeted Maximum
Likelihood Estimation for Time-to-Event Outcomes.” *Biometrics* 76 (3):
722–33. <https://doi.org/10.1111/biom.13172>.

</div>

<div id="ref-callawaySantanna2021" class="csl-entry">

Callaway, Brantly, and Pedro H. C. Sant’Anna. 2021.
“Difference-in-Differences with Multiple Time Periods.” *Journal of
Econometrics* 225 (2): 200–230.

</div>

<div id="ref-cavanagh1998rank" class="csl-entry">

Cavanagh, Christopher, and Robert P. Sherman. 1998. “Rank Estimators for
Monotonic Index Models.” *Journal of Econometrics* 84 (2): 351–81.
<https://doi.org/10.1016/S0304-4076(97)00090-0>.

</div>

<div id="ref-changLin2011libsvm" class="csl-entry">

Chang, Chih-Chung, and Chih-Jen Lin. 2011. “LIBSVM: A Library for
Support Vector Machines.” *ACM Transactions on Intelligent Systems and
Technology* 2 (3): 27:1–27. <https://doi.org/10.1145/1961189.1961199>.

</div>

<div id="ref-chen2002rank" class="csl-entry">

Chen, Songnian. 2002. “Rank Estimation of Transformation Models.”
*Econometrica* 70 (4): 1683–97.
<https://doi.org/10.1111/1468-0262.00347>.

</div>

<div id="ref-chenGuestrin2016xgboost" class="csl-entry">

Chen, Tianqi, and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree
Boosting System.” *Proceedings of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (KDD ’16)*, 785–94.
<https://doi.org/10.1145/2939672.2939785>.

</div>

<div id="ref-chernozhukov2018double" class="csl-entry">

Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, et al. 2018.
“Double/Debiased Machine Learning for Treatment and Structural
Parameters.” *The Econometrics Journal* 21 (1): C1–68.
<https://doi.org/10.1111/ectj.12097>.

</div>

<div id="ref-chernozhukov2018generic" class="csl-entry">

Chernozhukov, Victor, Mert Demirer, Esther Duflo, and Iván
Fernández-Val. 2018. *Generic Machine Learning Inference on
Heterogeneous Treatment Effects in Randomized Experiments*. No. 24678.
National Bureau of Economic Research. <https://doi.org/10.3386/w24678>.

</div>

<div id="ref-chernozhukovHongTamer2007" class="csl-entry">

Chernozhukov, Victor, Han Hong, and Elie Tamer. 2007. “Estimation and
Confidence Regions for Parameter Sets in Econometric Models.”
*Econometrica* 75 (5): 1243–84.
<https://doi.org/10.1111/j.1468-0262.2007.00794.x>.

</div>

<div id="ref-czado2019analyzing" class="csl-entry">

Czado, Claudia. 2019. *Analyzing Dependent Data with Vine Copulas: A
Practical Guide with R*. Vol. 222. Lecture Notes in Statistics.
Springer. <https://doi.org/10.1007/978-3-030-13785-4>.

</div>

<div id="ref-daley2003pointprocess" class="csl-entry">

Daley, D. J., and D. Vere-Jones. 2003. *An Introduction to the Theory of
Point Processes, Volume I: Elementary Theory and Methods*. 2nd ed.
Springer.

</div>

<div id="ref-davisonHinkley1997bootstrap" class="csl-entry">

Davison, Anthony C., and David V. Hinkley. 1997. *Bootstrap Methods and
Their Application*. Cambridge University Press.
<https://doi.org/10.1017/CBO9780511802843>.

</div>

<div id="ref-dekkers1989moment" class="csl-entry">

Dekkers, Arnold L. M., John H. J. Einmahl, and Laurens de Haan. 1989. “A
Moment Estimator for the Index of an Extreme-Value Distribution.” *The
Annals of Statistics* 17 (4): 1833–55.
<https://doi.org/10.1214/aos/1176347397>.

</div>

<div id="ref-diggle2003pointpatterns" class="csl-entry">

Diggle, Peter J. 2003. *Statistical Analysis of Spatial Point Patterns*.
2nd ed. Edward Arnold.

</div>

<div id="ref-dongLewbel2015simple" class="csl-entry">

Dong, Yingying, and Arthur Lewbel. 2015. “A Simple Estimator for Binary
Choice Models with Endogenous Regressors.” *Econometric Reviews* 34
(1–2): 82–105. <https://doi.org/10.1080/07474938.2014.944470>.

</div>

<div id="ref-donoho1994ideal" class="csl-entry">

Donoho, David L., and Iain M. Johnstone. 1994. “Ideal Spatial Adaptation
by Wavelet Shrinkage.” *Biometrika* 81 (3): 425–55.
<https://doi.org/10.1093/biomet/81.3.425>.

</div>

<div id="ref-doob2020affidavit" class="csl-entry">

Doob, Anthony N. 2020. *Affidavit of Anthony N. Doob*. Federal Court of
Canada, File T-539-20, Application Record Vol. 3 of 5, pp. 778–795.

</div>

<div id="ref-dwork2014algorithmic" class="csl-entry">

Dwork, Cynthia, and Aaron Roth. 2014. “The Algorithmic Foundations of
Differential Privacy.” *Foundations and Trends in Theoretical Computer
Science* 9 (3–4): 211–487. <https://doi.org/10.1561/0400000042>.

</div>

<div id="ref-efron1979bootstrap" class="csl-entry">

Efron, Bradley. 1979. “Bootstrap Methods: Another Look at the
Jackknife.” *The Annals of Statistics* 7 (1): 1–26.
<https://doi.org/10.1214/aos/1176344552>.

</div>

<div id="ref-efron1997improvements" class="csl-entry">

Efron, Bradley, and Robert Tibshirani. 1997. “Improvements on
Cross-Validation: The .632+ Bootstrap Method.” *Journal of the American
Statistical Association* 92 (438): 548–60.
<https://doi.org/10.1080/01621459.1997.10474007>.

</div>

<div id="ref-efronTibshirani1993introduction" class="csl-entry">

Efron, Bradley, and Robert J. Tibshirani. 1993. *An Introduction to the
Bootstrap*. Chapman; Hall. <https://doi.org/10.1201/9780429246593>.

</div>

<div id="ref-engle2002dcc" class="csl-entry">

Engle, Robert F. 2002. “Dynamic Conditional Correlation: A Simple Class
of Multivariate Generalized Autoregressive Conditional
Heteroskedasticity Models.” *Journal of Business & Economic Statistics*
20 (3): 339–50. <https://doi.org/10.1198/073500102288618487>.

</div>

<div id="ref-fan1991optimal" class="csl-entry">

Fan, Jianqing. 1991. “On the Optimal Rates of Convergence for
Nonparametric Deconvolution Problems.” *The Annals of Statistics* 19
(3): 1257–72. <https://doi.org/10.1214/aos/1176348248>.

</div>

<div id="ref-fanChenLin2005wss" class="csl-entry">

Fan, Rong-En, Pai-Hsuen Chen, and Chih-Jen Lin. 2005. “Working Set
Selection Using Second Order Information for Training Support Vector
Machines.” *Journal of Machine Learning Research* 6: 1889–918.

</div>

<div id="ref-fauzi2023kernel" class="csl-entry">

Fauzi, Rizky Reza, and Yoshihiko Maesono. 2023. *Statistical Inference
Based on Kernel Distribution Function Estimators*. SpringerBriefs in
Statistics. Springer. <https://doi.org/10.1007/978-981-99-1862-1>.

</div>

<div id="ref-ferroSegers2003inference" class="csl-entry">

Ferro, Christopher A. T., and Johan Segers. 2003. “Inference for
Clusters of Extreme Values.” *Journal of the Royal Statistical Society,
Series B* 65 (2): 545–56. <https://doi.org/10.1111/1467-9868.00401>.

</div>

<div id="ref-fieller1954interval" class="csl-entry">

Fieller, E. C. 1954. “Some Problems in Interval Estimation.” *Journal of
the Royal Statistical Society, Series B* 16 (2): 175–85.
<https://doi.org/10.1111/j.2517-6161.1954.tb00159.x>.

</div>

<div id="ref-finney1971probit" class="csl-entry">

Finney, D. J. 1971. *Probit Analysis*. 3rd ed. Cambridge University
Press.

</div>

<div id="ref-fisher1921probable" class="csl-entry">

Fisher, Ronald A. 1921. “On the ‘Probable Error’ of a Coefficient of
Correlation Deduced from a Small Sample.” *Metron* 1: 3–32.

</div>

<div id="ref-friedman2001greedy" class="csl-entry">

Friedman, Jerome H. 2001. “Greedy Function Approximation: A Gradient
Boosting Machine.” *The Annals of Statistics* 29 (5): 1189–232.
<https://doi.org/10.1214/aos/1013203451>.

</div>

<div id="ref-fuller1977some" class="csl-entry">

Fuller, Wayne A. 1977. “Some Properties of a Modification of the Limited
Information Estimator.” *Econometrica* 45 (4): 939–53.
<https://doi.org/10.2307/1912683>.

</div>

<div id="ref-garmanKlass1980estimation" class="csl-entry">

Garman, Mark B., and Michael J. Klass. 1980. “On the Estimation of
Security Price Volatilities from Historical Data.” *Journal of Business*
53 (1): 67–78. <https://doi.org/10.1086/296072>.

</div>

<div id="ref-geyer1992practical" class="csl-entry">

Geyer, Charles J. 1992. “Practical Markov Chain Monte Carlo.”
*Statistical Science* 7 (4): 473–83.
<https://doi.org/10.1214/ss/1177011137>.

</div>

<div id="ref-gibbonsChakraborti2011nonparametric" class="csl-entry">

Gibbons, Jean Dickinson, and Subhabrata Chakraborti. 2011.
*Nonparametric Statistical Inference*. 5th ed. Statistics: Textbooks and
Monographs. Chapman & Hall/CRC.

</div>

<div id="ref-goffman1961asylums" class="csl-entry">

Goffman, Erving. 1961. *Asylums: Essays on the Social Situation of
Mental Patients and Other Inmates*. Anchor Books.

</div>

<div id="ref-goodman1959acquaintances" class="csl-entry">

Goodman, A. W. 1959. “On Sets of Acquaintances and Strangers at Any
Party.” *The American Mathematical Monthly* 66 (9): 778–83.
<https://doi.org/10.1080/00029890.1959.11989408>.

</div>

<div id="ref-goodmanBacon2021did" class="csl-entry">

Goodman-Bacon, Andrew. 2021. “Difference-in-Differences with Variation
in Treatment Timing.” *Journal of Econometrics* 225 (2): 254–77.

</div>

<div id="ref-grassberger1983strangeness" class="csl-entry">

Grassberger, Peter, and Itamar Procaccia. 1983. “Measuring the
Strangeness of Strange Attractors.” *Physica D: Nonlinear Phenomena* 9
(1–2): 189–208. <https://doi.org/10.1016/0167-2789(83)90298-1>.

</div>

<div id="ref-greenacre1984correspondence" class="csl-entry">

Greenacre, Michael J. 1984. *Theory and Applications of Correspondence
Analysis*. Academic Press.

</div>

<div id="ref-gretton2012kernel" class="csl-entry">

Gretton, Arthur, Karsten M. Borgwardt, Malte J. Rasch, Bernhard
Schölkopf, and Alexander Smola. 2012. “A Kernel Two-Sample Test.”
*Journal of Machine Learning Research* 13: 723–73.
<https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf>.

</div>

<div id="ref-gruberVanDerLaan2010" class="csl-entry">

Gruber, Susan, and Mark J. van der Laan. 2010. “A Targeted Maximum
Likelihood Estimator of a Causal Effect on a Bounded Continuous
Outcome.” *International Journal of Biostatistics* 6 (1): Article 26.
<https://doi.org/10.2202/1557-4679.1260>.

</div>

<div id="ref-hahn1998propensity" class="csl-entry">

Hahn, Jinyong. 1998. “On the Role of the Propensity Score in Efficient
Semiparametric Estimation of Average Treatment Effects.” *Econometrica*
66 (2): 315–31. <https://doi.org/10.2307/2998560>.

</div>

<div id="ref-han1987nonparametric" class="csl-entry">

Han, Aaron K. 1987. “Non-Parametric Analysis of a Generalized Regression
Model: The Maximum Rank Correlation Estimator.” *Journal of
Econometrics* 35 (2–3): 303–16.
<https://doi.org/10.1016/0304-4076(87)90030-3>.

</div>

<div id="ref-han1978nonnegative" class="csl-entry">

Han, Te Sun. 1978. “Nonnegative Entropy Measures of Multivariate
Symmetric Correlations.” *Information and Control* 36 (2): 133–56.
<https://doi.org/10.1016/S0019-9958(78)90275-9>.

</div>

<div id="ref-hartley1962multiple" class="csl-entry">

Hartley, H. O. 1962. “Multiple Frame Surveys.” *Proceedings of the
Social Statistics Section*, 203–6.

</div>

<div id="ref-hastie2009esl" class="csl-entry">

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. *The
Elements of Statistical Learning: Data Mining, Inference, and
Prediction*. 2nd ed. Springer Series in Statistics. Springer.
<https://doi.org/10.1007/978-0-387-84858-7>.

</div>

<div id="ref-hedges2010robust" class="csl-entry">

Hedges, Larry V., Elizabeth Tipton, and Matthew C. Johnson. 2010.
“Robust Variance Estimation in Meta-Regression with Dependent Effect
Size Estimates.” *Research Synthesis Methods* 1 (1): 39–65.
<https://doi.org/10.1002/jrsm.5>.

</div>

<div id="ref-higuchi1988approach" class="csl-entry">

Higuchi, T. 1988. “Approach to an Irregular Time Series on the Basis of
the Fractal Theory.” *Physica D: Nonlinear Phenomena* 31 (2): 277–83.
<https://doi.org/10.1016/0167-2789(88)90081-4>.

</div>

<div id="ref-hill1975simple" class="csl-entry">

Hill, Bruce M. 1975. “A Simple General Approach to Inference about the
Tail of a Distribution.” *The Annals of Statistics* 3 (5): 1163–74.
<https://doi.org/10.1214/aos/1176343247>.

</div>

<div id="ref-hirano2009asymptotics" class="csl-entry">

Hirano, Keisuke, and Jack R. Porter. 2009. “Asymptotics for Statistical
Treatment Rules.” *Econometrica* 77 (5): 1683–701.
<https://doi.org/10.3982/ECTA6630>.

</div>

<div id="ref-horowitz1992smoothed" class="csl-entry">

Horowitz, Joel L. 1992. “A Smoothed Maximum Score Estimator for the
Binary Response Model.” *Econometrica* 60 (3): 505–31.
<https://doi.org/10.2307/2951582>.

</div>

<div id="ref-horowitz1996distribution" class="csl-entry">

Horowitz, Joel L. 1996. “Semiparametric Estimation of a Regression Model
with an Unknown Transformation of the Dependent Variable.”
*Econometrica* 64 (1): 103–37. <https://doi.org/10.2307/2171926>.

</div>

<div id="ref-horowitz2009semiparametric" class="csl-entry">

Horowitz, Joel L. 2009. *Semiparametric and Nonparametric Methods in
Econometrics*. Springer Series in Statistics. Springer.
<https://doi.org/10.1007/978-0-387-92870-8>.

</div>

<div id="ref-horowitzMarkatou1996semiparametric" class="csl-entry">

Horowitz, Joel L., and Marianthi Markatou. 1996. “Semiparametric
Estimation of Regression Models for Panel Data.” *The Review of Economic
Studies* 63 (1): 145–68. <https://doi.org/10.2307/2298119>.

</div>

<div id="ref-hosking1980multivariate" class="csl-entry">

Hosking, J. R. M. 1980. “The Multivariate Portmanteau Statistic.”
*Journal of the American Statistical Association* 75 (371): 602–8.
<https://doi.org/10.1080/01621459.1980.10477520>.

</div>

<div id="ref-hosking1990lmoments" class="csl-entry">

Hosking, J. R. M. 1990. “L-Moments: Analysis and Estimation of
Distributions Using Linear Combinations of Order Statistics.” *Journal
of the Royal Statistical Society, Series B* 52 (1): 105–24.
<https://doi.org/10.1111/j.2517-6161.1990.tb01775.x>.

</div>

<div id="ref-hoskingWallis1987parameter" class="csl-entry">

Hosking, J. R. M., and J. R. Wallis. 1987. “Parameter and Quantile
Estimation for the Generalized Pareto Distribution.” *Technometrics* 29
(3): 339–49. <https://doi.org/10.1080/00401706.1987.10488243>.

</div>

<div id="ref-hoskingWallisWood1985estimation" class="csl-entry">

Hosking, J. R. M., J. R. Wallis, and E. F. Wood. 1985. “Estimation of
the Generalized Extreme-Value Distribution by the Method of
Probability-Weighted Moments.” *Technometrics* 27 (3): 251–61.
<https://doi.org/10.1080/00401706.1985.10488049>.

</div>

<div id="ref-hubbard2000nonparametric" class="csl-entry">

Hubbard, Alan E., Mark J. van der Laan, and James M. Robins. 2000.
“Nonparametric Locally Efficient Estimation of the Treatment Specific
Survival Distribution with Right Censored Data and Covariates in
Observational Studies.” In *Statistical Models in Epidemiology, the
Environment, and Clinical Trials*, edited by M. Elizabeth Halloran and
Donald Berry. Springer. <https://doi.org/10.1007/978-1-4612-1284-3_3>.

</div>

<div id="ref-huber1973robust" class="csl-entry">

Huber, Peter J. 1973. “Robust Regression: Asymptotics, Conjectures and
Monte Carlo.” *The Annals of Statistics* 1 (5): 799–821.
<https://doi.org/10.1214/aos/1176342503>.

</div>

<div id="ref-ichimura1993semiparametric" class="csl-entry">

Ichimura, Hidehiko. 1993. “Semiparametric Least Squares (SLS) and
Weighted SLS Estimation of Single-Index Models.” *Journal of
Econometrics* 58 (1–2): 71–120.
<https://doi.org/10.1016/0304-4076(93)90114-K>.

</div>

<div id="ref-imaiKeeleYamamoto2010mediation" class="csl-entry">

Imai, Kosuke, Luke Keele, and Teppei Yamamoto. 2010. “Identification,
Inference and Sensitivity Analysis for Causal Mediation Effects.”
*Statistical Science* 25 (1): 51–71.
<https://doi.org/10.1214/10-STS321>.

</div>

<div id="ref-imbens2003sensitivity" class="csl-entry">

Imbens, Guido W. 2003. “Sensitivity to Exogeneity Assumptions in Program
Evaluation.” *American Economic Review* 93 (2): 126–32.
<https://doi.org/10.1257/000282803321946921>.

</div>

<div id="ref-imbens1994late" class="csl-entry">

Imbens, Guido W., and Joshua D. Angrist. 1994. “Identification and
Estimation of Local Average Treatment Effects.” *Econometrica* 62 (2):
467–75. <https://doi.org/10.2307/2951620>.

</div>

<div id="ref-imbensManski2004confidence" class="csl-entry">

Imbens, Guido W., and Charles F. Manski. 2004. “Confidence Intervals for
Partially Identified Parameters.” *Econometrica* 72 (6): 1845–57.
<https://doi.org/10.1111/j.1468-0262.2004.00555.x>.

</div>

<div id="ref-jacquez1996knn" class="csl-entry">

Jacquez, Geoffrey M. 1996. “A k Nearest Neighbour Test for Space-Time
Interaction.” *Statistics in Medicine* 15 (18): 1935–49.
[https://doi.org/10.1002/(SICI)1097-0258(19960930)15:18\<1935::AID-SIM406\>3.0.CO;2-I](https://doi.org/10.1002/(SICI)1097-0258(19960930)15:18<1935::AID-SIM406>3.0.CO;2-I).

</div>

<div id="ref-karwa2018finite" class="csl-entry">

Karwa, Vishesh, and Salil Vadhan. 2017. *Finite Sample Differentially
Private Confidence Intervals*. <https://arxiv.org/abs/1711.03908>.

</div>

<div id="ref-kleinSpady1993efficient" class="csl-entry">

Klein, Roger W., and Richard H. Spady. 1993. “An Efficient
Semiparametric Estimator for Binary Response Models.” *Econometrica* 61
(2): 387–421. <https://doi.org/10.2307/2951556>.

</div>

<div id="ref-kooremanMelenberg1989maximum" class="csl-entry">

Kooreman, Peter, and Bertrand Melenberg. 1989. *Maximum Score Estimation
in the Ordered Response Model*. Discussion Paper Nos. 1989-48. Tilburg
University, Center for Economic Research.

</div>

<div id="ref-kosorok2008introduction" class="csl-entry">

Kosorok, Michael R. 2008. *Introduction to Empirical Processes and
Semiparametric Inference*. Springer Series in Statistics. Springer.
<https://doi.org/10.1007/978-0-387-74978-5>.

</div>

<div id="ref-kunzel2019metalearners" class="csl-entry">

Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 2019.
“Metalearners for Estimating Heterogeneous Treatment Effects Using
Machine Learning.” *Proceedings of the National Academy of Sciences* 116
(10): 4156–65. <https://doi.org/10.1073/pnas.1804597116>.

</div>

<div id="ref-vanderlaan2007superlearner" class="csl-entry">

Laan, Mark J. van der, Eric C. Polley, and Alan E. Hubbard. 2007. “Super
Learner.” *Statistical Applications in Genetics and Molecular Biology* 6
(1).

</div>

<div id="ref-vanderLaanRubin2006tmle" class="csl-entry">

Laan, Mark J. van der, and Daniel Rubin. 2006. “Targeted Maximum
Likelihood Learning.” *International Journal of Biostatistics* 2 (1):
Article 11. <https://doi.org/10.2202/1557-4679.1043>.

</div>

<div id="ref-lavine1992polya" class="csl-entry">

Lavine, Michael. 1992. “Some Aspects of Polya Tree Distributions for
Statistical Modelling.” *The Annals of Statistics* 20 (3): 1222–35.
<https://doi.org/10.1214/aos/1176348767>.

</div>

<div id="ref-lewbel2000semiparametric" class="csl-entry">

Lewbel, Arthur. 2000. “Semiparametric Qualitative Response Model
Estimation with Unknown Heteroscedasticity or Instrumental Variables.”
*Journal of Econometrics* 97 (1): 145–77.
<https://doi.org/10.1016/S0304-4076(00)00015-4>.

</div>

<div id="ref-lohrRao2000dualframe" class="csl-entry">

Lohr, Sharon L., and J. N. K. Rao. 2000. “Inference from Dual Frame
Surveys.” *Journal of the American Statistical Association* 95 (449):
271–80. <https://doi.org/10.1080/01621459.2000.10473920>.

</div>

<div id="ref-lord1980applications" class="csl-entry">

Lord, Frederic M. 1980. *Applications of Item Response Theory to
Practical Testing Problems*. Lawrence Erlbaum Associates.

</div>

<div id="ref-lu2017bracken" class="csl-entry">

Lu, Jennifer, Florian P. Breitwieser, Peter Thielen, and Steven L.
Salzberg. 2017. “Bracken: Estimating Species Abundance in Metagenomics
Data.” *PeerJ Computer Science* 3: e104.
<https://doi.org/10.7717/peerj-cs.104>.

</div>

<div id="ref-lumley2010complex" class="csl-entry">

Lumley, Thomas. 2010. *Complex Surveys: A Guide to Analysis Using R*.
Wiley. <https://doi.org/10.1002/9780470580066>.

</div>

<div id="ref-mackay2003information" class="csl-entry">

MacKay, David J. C. 2003. *Information Theory, Inference, and Learning
Algorithms*. Cambridge University Press.

</div>

<div id="ref-mahdi2020portes" class="csl-entry">

Mahdi, Esam. 2020. *Portes: An R Package for Portmanteau Tests in Time
Series Models*. <https://arxiv.org/abs/2005.00931>.

</div>

<div id="ref-manski1985semiparametric" class="csl-entry">

Manski, Charles F. 1985. “Semiparametric Analysis of Discrete Response:
Asymptotic Properties of the Maximum Score Estimator.” *Journal of
Econometrics* 27 (3): 313–33.
<https://doi.org/10.1016/0304-4076(85)90009-0>.

</div>

<div id="ref-manski1987semiparametric" class="csl-entry">

Manski, Charles F. 1987. “Semiparametric Analysis of Random Effects
Linear Models from Binary Panel Data.” *Econometrica* 55 (2): 357–62.
<https://doi.org/10.2307/1913020>.

</div>

<div id="ref-manski1990nonparametric" class="csl-entry">

Manski, Charles F. 1990. “Nonparametric Bounds on Treatment Effects.”
*American Economic Review: Papers and Proceedings* 80 (2): 319–23.

</div>

<div id="ref-manskiTamer2002inference" class="csl-entry">

Manski, Charles F., and Elie Tamer. 2002. “Inference on Regressions with
Interval Data on a Regressor or Outcome.” *Econometrica* 70 (2): 519–46.
<https://doi.org/10.1111/1468-0262.00294>.

</div>

<div id="ref-marple1987digital" class="csl-entry">

Marple, S. Lawrence. 1987. *Digital Spectral Analysis with
Applications*. Prentice-Hall.

</div>

<div id="ref-matthews1975comparison" class="csl-entry">

Matthews, Brian W. 1975. “Comparison of the Predicted and Observed
Secondary Structure of T4 Phage Lysozyme.” *Biochimica Et Biophysica
Acta (BBA) — Protein Structure* 405 (2): 442–51.
<https://doi.org/10.1016/0005-2795(75)90109-9>.

</div>

<div id="ref-mcallester2020formal" class="csl-entry">

McAllester, David, and Karl Stratos. 2020. “Formal Limitations on the
Measurement of Mutual Information.” *Proceedings of the Twenty Third
International Conference on Artificial Intelligence and Statistics*,
Proceedings of machine learning research, vol. 108: 875–84.

</div>

<div id="ref-mcgraw1996forming" class="csl-entry">

McGraw, Kenneth O., and S. P. Wong. 1996. “Forming Inferences about Some
Intraclass Correlation Coefficients.” *Psychological Methods* 1 (1):
30–46. <https://doi.org/10.1037/1082-989X.1.1.30>.

</div>

<div id="ref-melenbergVanSoest1996parametric" class="csl-entry">

Melenberg, Bertrand, and Arthur van Soest. 1996. “Measuring the Costs of
Children: Parametric and Semiparametric Estimators.” *Statistica
Neerlandica* 50 (1): 171–92.
<https://doi.org/10.1111/j.1467-9574.1996.tb01486.x>.

</div>

<div id="ref-mislevy1986bayes" class="csl-entry">

Mislevy, Robert J. 1986. “Bayes Modal Estimation in Item Response
Models.” *Psychometrika* 51 (2): 177–95.
<https://doi.org/10.1007/BF02293979>.

</div>

<div id="ref-mogstadSantosTorgovitsky2018" class="csl-entry">

Mogstad, Magne, Andres Santos, and Alexander Torgovitsky. 2018. “Using
Instrumental Variables for Inference about Policy Relevant Treatment
Parameters.” *Econometrica* 86 (5): 1589–619.
<https://doi.org/10.3982/ECTA15463>.

</div>

<div id="ref-mooreVanDerLaan2009censored" class="csl-entry">

Moore, Kelly L., and Mark J. van der Laan. 2009. “Increasing Power in
Randomized Trials with Right Censored Outcomes Through Covariate
Adjustment.” *Journal of Biopharmaceutical Statistics* 19 (6): 1099–131.
<https://doi.org/10.1080/10543400903243017>.

</div>

<div id="ref-moran1950notes" class="csl-entry">

Moran, P. A. P. 1950. “Notes on Continuous Stochastic Phenomena.”
*Biometrika* 37 (1/2): 17–23. <https://doi.org/10.2307/2332142>.

</div>

<div id="ref-naveau2009modelling" class="csl-entry">

Naveau, Philippe, Armelle Guillou, Daniel Cooley, and Jean Diebolt.
2009. “Modelling Pairwise Dependence of Maxima in Space.” *Biometrika*
96 (1): 1–17. <https://doi.org/10.1093/biomet/asn044>.

</div>

<div id="ref-nenadicGreenacre2007ca" class="csl-entry">

Nenadic, Oleg, and Michael Greenacre. 2007. “Correspondence Analysis in
R, with Two- and Three-Dimensional Graphics: The Ca Package.” *Journal
of Statistical Software* 20 (3): 1–13.
<https://doi.org/10.18637/jss.v020.i03>.

</div>

<div id="ref-neweyStoker1993efficiency" class="csl-entry">

Newey, Whitney K., and Thomas M. Stoker. 1993. “Efficiency of Weighted
Average Derivative Estimators and Index Models.” *Econometrica* 61 (5):
1199–223. <https://doi.org/10.2307/2951498>.

</div>

<div id="ref-nie2021quasi" class="csl-entry">

Nie, Xinkun, and Stefan Wager. 2021. “Quasi-Oracle Estimation of
Heterogeneous Treatment Effects.” *Biometrika* 108 (2): 299–319.
<https://doi.org/10.1093/biomet/asaa076>.

</div>

<div id="ref-northrop2015efficient" class="csl-entry">

Northrop, Paul J. 2015. “An Efficient Semiparametric Maxima Estimator of
the Extremal Index.” *Extremes* 18 (4): 585–603.
<https://doi.org/10.1007/s10687-015-0221-5>.

</div>

<div id="ref-nunnally1994psychometric" class="csl-entry">

Nunnally, Jum C., and Ira H. Bernstein. 1994. *Psychometric Theory*. 3rd
ed. McGraw-Hill.

</div>

<div id="ref-parkinson1980extreme" class="csl-entry">

Parkinson, Michael. 1980. “The Extreme Value Method for Estimating the
Variance of the Rate of Return.” *Journal of Business* 53 (1): 61–65.
<https://doi.org/10.1086/296071>.

</div>

<div id="ref-parzen1962estimation" class="csl-entry">

Parzen, Emanuel. 1962. “On Estimation of a Probability Density Function
and Mode.” *The Annals of Mathematical Statistics* 33 (3): 1065–76.
<https://doi.org/10.1214/aoms/1177704472>.

</div>

<div id="ref-paule1982consensus" class="csl-entry">

Paule, Robert C., and John Mandel. 1982. “Consensus Values and Weighting
Factors.” *Journal of Research of the National Bureau of Standards* 87
(5): 377–85. <https://doi.org/10.6028/jres.087.022>.

</div>

<div id="ref-pearson1897spurious" class="csl-entry">

Pearson, Karl. 1897. “Mathematical Contributions to the Theory of
Evolution.—on a Form of Spurious Correlation Which May Arise When
Indices Are Used in the Measurement of Organs.” *Proceedings of the
Royal Society of London* 60: 489–98.
<https://doi.org/10.1098/rspl.1896.0076>.

</div>

<div id="ref-pedroni1999critical" class="csl-entry">

Pedroni, Peter. 1999. “Critical Values for Cointegration Tests in
Heterogeneous Panels with Multiple Regressors.” *Oxford Bulletin of
Economics and Statistics* 61 (S1): 653–70.
<https://doi.org/10.1111/1468-0084.61.s1.14>.

</div>

<div id="ref-peng1994mosaic" class="csl-entry">

Peng, C.-K., S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and A.
L. Goldberger. 1994. “Mosaic Organization of DNA Nucleotides.” *Physical
Review E* 49 (2): 1685–89. <https://doi.org/10.1103/PhysRevE.49.1685>.

</div>

<div id="ref-percival2000wavelet" class="csl-entry">

Percival, Donald B., and Andrew T. Walden. 2000. *Wavelet Methods for
Time Series Analysis*. Cambridge Series in Statistical and Probabilistic
Mathematics. Cambridge University Press.

</div>

<div id="ref-pickands1975statistical" class="csl-entry">

Pickands, James. 1975. “Statistical Inference Using Extreme Order
Statistics.” *The Annals of Statistics* 3 (1): 119–31.
<https://doi.org/10.1214/aos/1176343003>.

</div>

<div id="ref-pincus1991approximate" class="csl-entry">

Pincus, Steven M. 1991. “Approximate Entropy as a Measure of System
Complexity.” *Proceedings of the National Academy of Sciences* 88 (6):
2297–301. <https://doi.org/10.1073/pnas.88.6.2297>.

</div>

<div id="ref-poole1985spatial" class="csl-entry">

Poole, Keith T., and Howard Rosenthal. 1985. “A Spatial Model for
Legislative Roll Call Analysis.” *American Journal of Political Science*
29 (2): 357–84. <https://doi.org/10.2307/2111172>.

</div>

<div id="ref-powellStockStoker1989semiparametric" class="csl-entry">

Powell, James L., James H. Stock, and Thomas M. Stoker. 1989.
“Semiparametric Estimation of Index Coefficients.” *Econometrica* 57
(6): 1403–30. <https://doi.org/10.2307/1913713>.

</div>

<div id="ref-quenouille1949approximate" class="csl-entry">

Quenouille, Maurice H. 1949. “Approximate Tests of Correlation in
Time-Series.” *Journal of the Royal Statistical Society, Series B* 11
(1): 68–84. <https://doi.org/10.1111/j.2517-6161.1949.tb00023.x>.

</div>

<div id="ref-radziszowski2024ramsey" class="csl-entry">

Radziszowski, Stanisław P. 2024. *Small Ramsey Numbers*. Electronic
Journal of Combinatorics, Dynamic Survey DS1, revision 17.
<https://doi.org/10.37236/21>.

</div>

<div id="ref-ramdas2017wasserstein" class="csl-entry">

Ramdas, Aaditya, Nicolás García Trillos, and Marco Cuturi. 2017. “On
Wasserstein Two-Sample Testing and Related Families of Nonparametric
Tests.” *Entropy* 19 (2): 47. <https://doi.org/10.3390/e19020047>.

</div>

<div id="ref-ramsay2005functional" class="csl-entry">

Ramsay, James O., and Bernard W. Silverman. 2005. *Functional Data
Analysis*. 2nd ed. Springer. <https://doi.org/10.1007/b98888>.

</div>

<div id="ref-ramsey1969reset" class="csl-entry">

Ramsey, J. B. 1969. “Tests for Specification Errors in Classical Linear
Least-Squares Regression Analysis.” *Journal of the Royal Statistical
Society, Series B* 31 (2): 350–71.
<https://doi.org/10.1111/j.2517-6161.1969.tb00796.x>.

</div>

<div id="ref-rangayyan2024biomedical" class="csl-entry">

Rangayyan, Rangaraj M., and Sridhar Krishnan. 2024. *Biomedical Signal
Analysis*. Third. IEEE Press Series in Biomedical Engineering. IEEE
Press / Wiley.

</div>

<div id="ref-richman2000physiological" class="csl-entry">

Richman, Joshua S., and J. Randall Moorman. 2000. “Physiological
Time-Series Analysis Using Approximate Entropy and Sample Entropy.”
*American Journal of Physiology-Heart and Circulatory Physiology* 278
(6): H2039–49. <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>.

</div>

<div id="ref-ripley1977modelling" class="csl-entry">

Ripley, Brian D. 1977. “Modelling Spatial Patterns.” *Journal of the
Royal Statistical Society, Series B* 39 (2): 172–212.
<https://doi.org/10.1111/j.2517-6161.1977.tb01615.x>.

</div>

<div id="ref-robins1994estimation" class="csl-entry">

Robins, James M., Andrea Rotnitzky, and Lue Ping Zhao. 1994. “Estimation
of Regression Coefficients When Some Regressors Are Not Always
Observed.” *Journal of the American Statistical Association* 89 (427):
846–66. <https://doi.org/10.1080/01621459.1994.10476818>.

</div>

<div id="ref-rosenstein1993practical" class="csl-entry">

Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. 1993. “A
Practical Method for Calculating Largest Lyapunov Exponents from Small
Data Sets.” *Physica D: Nonlinear Phenomena* 65 (1–2): 117–34.
<https://doi.org/10.1016/0167-2789(93)90009-P>.

</div>

<div id="ref-rousseeuwCroux1993alternatives" class="csl-entry">

Rousseeuw, Peter J., and Christophe Croux. 1993. “Alternatives to the
Median Absolute Deviation.” *Journal of the American Statistical
Association* 88 (424): 1273–83.
<https://doi.org/10.1080/01621459.1993.10476408>.

</div>

<div id="ref-rousseeuwYohai1984robust" class="csl-entry">

Rousseeuw, Peter J., and Victor J. Yohai. 1984. “Robust Regression by
Means of S-Estimators.” In *Robust and Nonlinear Time Series Analysis*,
vol. 26. Lecture Notes in Statistics. Springer.
<https://doi.org/10.1007/978-1-4615-7821-5_15>.

</div>

<div id="ref-ruhela2026dlrm" class="csl-entry">

Ruhela, Vansh Singh. 2026. “The MRM Framework: A Multi-Source
Statistical Foundation for Canadian Carceral, Police, and Oversight
Data, Implemented as MRM Modules in MORIE.” Unpublished manuscript.

</div>

<div id="ref-samejima1969estimation" class="csl-entry">

Samejima, Fumiko. 1969. *Estimation of Latent Ability Using a Response
Pattern of Graded Scores*. Psychometrika Monograph Supplement, No. 17.
Psychometric Society.

</div>

<div id="ref-samejima1973reliability" class="csl-entry">

Samejima, Fumiko. 1973. “A Comment on
<span class="nocase">Birnbaum’s</span> Three-Parameter Logistic Model in
the Latent Trait Theory.” *Psychometrika* 38 (2): 221–33.
<https://doi.org/10.1007/BF02291116>.

</div>

<div id="ref-santannaZhao2020drdid" class="csl-entry">

Sant’Anna, Pedro H. C., and Jun Zhao. 2020. “Doubly Robust
Difference-in-Differences Estimators.” *Journal of Econometrics* 219
(1): 101–22. <https://doi.org/10.1016/j.jeconom.2020.06.003>.

</div>

<div id="ref-sargan1958estimation" class="csl-entry">

Sargan, John D. 1958. “The Estimation of Economic Relationships Using
Instrumental Variables.” *Econometrica* 26 (3): 393–415.
<https://doi.org/10.2307/1907619>.

</div>

<div id="ref-schabenberger2005spatial" class="csl-entry">

Schabenberger, Oliver, and Carol A. Gotway. 2005. *Statistical Methods
for Spatial Data Analysis*. Chapman; Hall/CRC.

</div>

<div id="ref-sen1968estimates" class="csl-entry">

Sen, Pranab Kumar. 1968. “Estimates of the Regression Coefficient Based
on Kendall’s Tau.” *Journal of the American Statistical Association* 63
(324): 1379–89. <https://doi.org/10.1080/01621459.1968.10480934>.

</div>

<div id="ref-shealyStout1993sibtest" class="csl-entry">

Shealy, Robin, and William Stout. 1993. “A Model-Based Standardization
Approach That Separates True Bias/DIF from Group Ability Differences and
Detects Test Bias/DTF as Well as Item Bias/DIF.” *Psychometrika* 58 (2):
159–94. <https://doi.org/10.1007/BF02294572>.

</div>

<div id="ref-sherman1993limiting" class="csl-entry">

Sherman, Robert P. 1993. “The Limiting Distribution of the Maximum Rank
Correlation Estimator.” *Econometrica* 61 (1): 123–37.
<https://doi.org/10.2307/2951780>.

</div>

<div id="ref-shortbrantingham2008" class="csl-entry">

Short, M. B., M. R. D’Orsogna, V. B. Pasour, et al. 2008. “A Statistical
Model of Criminal Behavior.” *Mathematical Models and Methods in Applied
Sciences* 18 (Suppl.): 1249–67.

</div>

<div id="ref-shrout1979intraclass" class="csl-entry">

Shrout, Patrick E., and Joseph L. Fleiss. 1979. “Intraclass
Correlations: Uses in Assessing Rater Reliability.” *Psychological
Bulletin* 86 (2): 420–28. <https://doi.org/10.1037/0033-2909.86.2.420>.

</div>

<div id="ref-silverman1986density" class="csl-entry">

Silverman, Bernard W. 1986. *Density Estimation for Statistics and Data
Analysis*. Chapman; Hall. <https://doi.org/10.1201/9781315140919>.

</div>

<div id="ref-sloczynski2022interpreting" class="csl-entry">

Słoczyński, Tymon. 2022. “Interpreting OLS Estimands When Treatment
Effects Are Heterogeneous.” *Review of Economics and Statistics* 104
(3): 501–9. <https://doi.org/10.1162/rest_a_00953>.

</div>

<div id="ref-smithWeissman1994estimating" class="csl-entry">

Smith, Richard L., and Ishay Weissman. 1994. “Estimating the Extremal
Index.” *Journal of the Royal Statistical Society, Series B* 56 (3):
515–28. <https://doi.org/10.1111/j.2517-6161.1994.tb01997.x>.

</div>

<div id="ref-spielman1993transmission" class="csl-entry">

Spielman, Richard S., Ralph E. McGinnis, and Warren J. Ewens. 1993.
“Transmission Test for Linkage Disequilibrium: The Insulin Gene Region
and Insulin-Dependent Diabetes Mellitus (IDDM).” *American Journal of
Human Genetics* 52 (3): 506–16.

</div>

<div id="ref-sprottdoob2020covid" class="csl-entry">

Sprott, Jane B., and Anthony N. Doob. 2020a. *Is There Clear Evidence
That COVID-19 Was the Cause of Problems with the Operation of CSC’s
Structured Intervention Units?* Centre for Criminology; Sociolegal
Studies, University of Toronto.

</div>

<div id="ref-sprottdoob2020operation" class="csl-entry">

Sprott, Jane B., and Anthony N. Doob. 2020b. *Understanding the
Operation of Correctional Service Canada’s Structured Intervention
Units: Some Preliminary Findings*. Centre for Criminology; Sociolegal
Studies, University of Toronto.

</div>

<div id="ref-sprottdoob2021torture" class="csl-entry">

Sprott, Jane B., and Anthony N. Doob. 2021. *Solitary Confinement,
Torture, and Canada’s Structured Intervention Units*. Centre for
Criminology; Sociolegal Studies, University of Toronto.

</div>

<div id="ref-sprottdoobiftene2021iedm" class="csl-entry">

Sprott, Jane B., Anthony N. Doob, and Adelina Iftene. 2021. *Do
Independent External Decision Makers Ensure That “an Inmate’s
Confinement in a Structured Intervention Unit Is to End as Soon as
Possible”? \[Corrections and Conditional Release Act, Section 33\]*.
Schulich School of Law, Dalhousie University.

</div>

<div id="ref-stanley2011enumerative" class="csl-entry">

Stanley, Richard P. 2011. *Enumerative Combinatorics, Volume 1*. 2nd ed.
Vol. 49. Cambridge Studies in Advanced Mathematics. Cambridge University
Press.

</div>

<div id="ref-stoye2009more" class="csl-entry">

Stoye, Jörg. 2009. “More on Confidence Intervals for Partially
Identified Parameters.” *Econometrica* 77 (4): 1299–315.
<https://doi.org/10.3982/ECTA7347>.

</div>

<div id="ref-sun2021estimating" class="csl-entry">

Sun, Liyang, and Sarah Abraham. 2021. “Estimating Dynamic Treatment
Effects in Event Studies with Heterogeneous Treatment Effects.” *Journal
of Econometrics* 225 (2): 175–99.
<https://doi.org/10.1016/j.jeconom.2020.09.006>.

</div>

<div id="ref-taskforce1996heart" class="csl-entry">

Task Force of the European Society of Cardiology and the North American
Society of Pacing and Electrophysiology. 1996. “Heart Rate Variability:
Standards of Measurement, Physiological Interpretation, and Clinical
Use.” *Circulation* 93 (5): 1043–65.
<https://doi.org/10.1161/01.CIR.93.5.1043>.

</div>

<div id="ref-theil1950rank" class="csl-entry">

Theil, Henri. 1950. “A Rank-Invariant Method of Linear and Polynomial
Regression Analysis.” *Proceedings of the Koninklijke Nederlandse
Akademie van Wetenschappen* 53: 386–92, 521–25, 1397–412.

</div>

<div id="ref-tipton2015small" class="csl-entry">

Tipton, Elizabeth. 2015. “Small Sample Adjustments for Robust Variance
Estimation with Meta-Regression.” *Psychological Methods* 20 (3):
375–93. <https://doi.org/10.1037/met0000011>.

</div>

<div id="ref-tsay2010analysis" class="csl-entry">

Tsay, Ruey S. 2010. *Analysis of Financial Time Series*. 3rd ed. Wiley.
<https://doi.org/10.1002/9780470644560>.

</div>

<div id="ref-un2015mandela" class="csl-entry">

United Nations General Assembly. 2015. *United Nations Standard Minimum
Rules for the Treatment of Prisoners (the Nelson Mandela Rules)*.
A/Res/70/175.

</div>

<div id="ref-vanderweele2017evalue" class="csl-entry">

VanderWeele, Tyler J., and Peng Ding. 2017. “Sensitivity Analysis in
Observational Research: Introducing the E-Value.” *Annals of Internal
Medicine* 167 (4): 268–74.

</div>

<div id="ref-vanraden2008efficient" class="csl-entry">

VanRaden, P. M. 2008. “Efficient Methods to Compute Genomic
Predictions.” *Journal of Dairy Science* 91 (11): 4414–23.
<https://doi.org/10.3168/jds.2007-0980>.

</div>

<div id="ref-veitch2020text" class="csl-entry">

Veitch, Victor, Dhanya Sridhar, and David M. Blei. 2020. *Adapting Text
Embeddings for Causal Inference*. <https://arxiv.org/abs/1905.12741>.

</div>

<div id="ref-veroniki2016methods" class="csl-entry">

Veroniki, Areti Angeliki, Dan Jackson, Wolfgang Viechtbauer, et al.
2016. “Methods to Estimate the Between-Study Variance and Its
Uncertainty in Meta-Analysis.” *Research Synthesis Methods* 7 (1):
55–79. <https://doi.org/10.1002/jrsm.1164>.

</div>

<div id="ref-viechtbauer2005bias" class="csl-entry">

Viechtbauer, Wolfgang. 2005. “Bias and Efficiency of Meta-Analytic
Variance Estimators in the Random-Effects Model.” *Journal of
Educational and Behavioral Statistics* 30 (3): 261–93.
<https://doi.org/10.3102/10769986030003261>.

</div>

<div id="ref-viechtbauer2010conducting" class="csl-entry">

Viechtbauer, Wolfgang. 2010. “Conducting Meta-Analyses in R with the
<span class="nocase">metafor</span> Package.” *Journal of Statistical
Software* 36 (3): 1–48. <https://doi.org/10.18637/jss.v036.i03>.

</div>

<div id="ref-viechtbauer2010outlier" class="csl-entry">

Viechtbauer, Wolfgang, and Mike W.-L. Cheung. 2010. “Outlier and
Influence Diagnostics for Meta-Analysis.” *Research Synthesis Methods* 1
(2): 112–25. <https://doi.org/10.1002/jrsm.11>.

</div>

<div id="ref-villani2009optimal" class="csl-entry">

Villani, Cédric. 2009. *Optimal Transport: Old and New*. Vol. 338.
Grundlehren Der Mathematischen Wissenschaften. Springer.
<https://doi.org/10.1007/978-3-540-71050-9>.

</div>

<div id="ref-wagerAthey2018" class="csl-entry">

Wager, Stefan, and Susan Athey. 2018. “Estimation and Inference of
Heterogeneous Treatment Effects Using Random Forests.” *Journal of the
American Statistical Association* 113 (523): 1228–42.
<https://doi.org/10.1080/01621459.2017.1319839>.

</div>

<div id="ref-wagerHastieEfron2014" class="csl-entry">

Wager, Stefan, Trevor Hastie, and Bradley Efron. 2014. “Confidence
Intervals for Random Forests: The Jackknife and the Infinitesimal
Jackknife.” *Journal of Machine Learning Research* 15: 1625–51.

</div>

<div id="ref-wallace2009csi" class="csl-entry">

Wallace, Marnie, John Turner, Anthony Matarazzo, and Colin Babyak. 2009.
*Measuring Crime in Canada: Introducing the Crime Severity Index and
Improvements to the Uniform Crime Reporting Survey*. 85-004-X.
Statistics Canada.

</div>

<div id="ref-warm1989weighted" class="csl-entry">

Warm, Thomas A. 1989. “Weighted Likelihood Estimation of Ability in Item
Response Theory.” *Psychometrika* 54 (3): 427–50.
<https://doi.org/10.1007/BF02294627>.

</div>

<div id="ref-watanabe1960information" class="csl-entry">

Watanabe, Satosi. 1960. “Information Theoretical Analysis of
Multivariate Correlation.” *IBM Journal of Research and Development* 4
(1): 66–82. <https://doi.org/10.1147/rd.41.0066>.

</div>

<div id="ref-welch1967use" class="csl-entry">

Welch, Peter D. 1967. “The Use of Fast Fourier Transform for the
Estimation of Power Spectra: A Method Based on Time Averaging over
Short, Modified Periodograms.” *IEEE Transactions on Audio and
Electroacoustics* 15 (2): 70–73.
<https://doi.org/10.1109/TAU.1967.1161901>.

</div>

<div id="ref-widrow1985adaptive" class="csl-entry">

Widrow, Bernard, and Samuel D. Stearns. 1985. *Adaptive Signal
Processing*. Prentice-Hall.

</div>

<div id="ref-williamson2021nonparametric" class="csl-entry">

Williamson, Brian D., Peter B. Gilbert, Marco Carone, and Noah Simon.
2021. “Nonparametric Variable Importance Assessment Using Machine
Learning Techniques.” *Biometrics* 77 (1): 9–22.
<https://doi.org/10.1111/biom.13392>.

</div>

<div id="ref-williamson2023general" class="csl-entry">

Williamson, Brian D., Peter B. Gilbert, Noah Simon, and Marco Carone.
2023. “A General Framework for Inference on Algorithm-Agnostic Variable
Importance.” *Journal of the American Statistical Association* 118
(543): 1645–58. <https://doi.org/10.1080/01621459.2021.2003200>.

</div>

<div id="ref-wooldridge2010econometric" class="csl-entry">

Wooldridge, Jeffrey M. 2010. *Econometric Analysis of Cross Section and
Panel Data*. 2nd ed. MIT Press.

</div>

<div id="ref-wooldridge2021twoway" class="csl-entry">

Wooldridge, Jeffrey M. 2021. *Two-Way Fixed Effects, the Two-Way Mundlak
Regression, and Difference-in-Differences Estimators*. No. 3906345.
SSRN. <https://doi.org/10.2139/ssrn.3906345>.

</div>

<div id="ref-xiao2016face" class="csl-entry">

Xiao, Luo, Vadim Zipunnikov, David Ruppert, and Ciprian Crainiceanu.
2016. “Fast Covariance Estimation for High-Dimensional Functional Data.”
*Statistics and Computing* 26 (1–2): 409–21.
<https://doi.org/10.1007/s11222-014-9485-x>.

</div>

<div id="ref-yang2019xlnet" class="csl-entry">

Yang, Zhilin, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan
Salakhutdinov, and Quoc V. Le. 2019. *XLNet: Generalized Autoregressive
Pretraining for Language Understanding*.
<https://arxiv.org/abs/1906.08237>.

</div>

<div id="ref-yohai1987high" class="csl-entry">

Yohai, Victor J. 1987. “High Breakdown-Point and High Efficiency Robust
Estimates for Regression.” *The Annals of Statistics* 15 (2): 642–56.
<https://doi.org/10.1214/aos/1176350366>.

</div>

<div id="ref-yohaiZamar1988high" class="csl-entry">

Yohai, Victor J., and Ruben H. Zamar. 1988. “High Breakdown-Point
Estimates of Regression by Means of the Minimization of an Efficient
Scale.” *Journal of the American Statistical Association* 83 (402):
406–13. <https://doi.org/10.1080/01621459.1988.10478611>.

</div>

<div id="ref-zandieh2026turboquant" class="csl-entry">

Zandieh, Amir, Majid Daliri, Milad Hadian, and Vahab Mirrokni. 2026.
“TurboQuant: Online Vector Quantization with Near-Optimal Distortion
Rate.” *International Conference on Learning Representations (ICLR)*.
<https://doi.org/10.48550/arXiv.2504.19874>.

</div>

<div id="ref-zhangMyklandAitSahalia2005tale" class="csl-entry">

Zhang, Lan, Per A. Mykland, and Yacine Aït-Sahalia. 2005. “A Tale of Two
Time Scales: Determining Integrated Volatility with Noisy High-Frequency
Data.” *Journal of the American Statistical Association* 100 (472):
1394–411. <https://doi.org/10.1198/016214505000000169>.

</div>

</div>
