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
Tsay [-@tsay2010analysis].

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

<!-- Written out in full so the bibliography is readable in the
     repository and on GitHub, which renders markdown without
     citeproc. `suppress-bibliography: true` in the front matter
     stops Pandoc emitting a second copy at build time.
     Regenerate after editing paper.bib:
       pandoc paper.md --citeproc --bibliography=paper.bib -t gfm -o - \
         | sed -n '/^# References/,$p'
     Last regenerated 2026-07-28. -->

Aitchison, John. 1986. *The Statistical Analysis of Compositional Data*.
Monographs on Statistics and Applied Probability. Chapman & Hall.

Austin, Peter C. 2009. “Balance Diagnostics for Comparing the
Distribution of Baseline Covariates Between Treatment Groups in
Propensity-Score Matched Samples.” *Statistics in Medicine* 28 (25):
3083–107. <https://doi.org/10.1002/sim.3697>.

Belsley, David A., Edwin Kuh, and Roy E. Welsch. 1980. *Regression
Diagnostics: Identifying Influential Data and Sources of Collinearity*.
John Wiley & Sons. <https://doi.org/10.1002/0471725153>.

Bettencourt, Luı́s M. A., José Lobo, Dirk Helbing, Christian Kühnert, and
Geoffrey B. West. 2007. “Growth, Innovation, Scaling, and the Pace of
Life in Cities.” *Proceedings of the National Academy of Sciences* 104
(17): 7301–6.

Bollerslev, Tim. 1990. “Modelling the Coherence in Short-Run Nominal
Exchange Rates: A Multivariate Generalized ARCH Model.” *The Review of
Economics and Statistics* 72 (3): 498–505.
<https://doi.org/10.2307/2109358>.

Breiman, Leo. 2001. “Random Forests.” *Machine Learning* 45 (1): 5–32.
<https://doi.org/10.1023/A:1010933404324>.

Breslow, Norman E. 1974. “Covariance Analysis of Censored Survival
Data.” *Biometrics* 30 (1): 89–99. <https://doi.org/10.2307/2529620>.

Burg, John Parker. 1975. “Maximum Entropy Spectral Analysis.” PhD
thesis, Stanford University.

Cavanagh, Christopher, and Robert P. Sherman. 1998. “Rank Estimators for
Monotonic Index Models.” *Journal of Econometrics* 84 (2): 351–81.
<https://doi.org/10.1016/S0304-4076(97)00090-0>.

Chang, Chih-Chung, and Chih-Jen Lin. 2011. “LIBSVM: A Library for
Support Vector Machines.” *ACM Transactions on Intelligent Systems and
Technology* 2 (3): 27:1–27. <https://doi.org/10.1145/1961189.1961199>.

Chen, Songnian. 2002. “Rank Estimation of Transformation Models.”
*Econometrica* 70 (4): 1683–97.
<https://doi.org/10.1111/1468-0262.00347>.

Chen, Tianqi, and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree
Boosting System.” *Proceedings of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (KDD ’16)*, 785–94.
<https://doi.org/10.1145/2939672.2939785>.

Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, et al. 2018.
“Double/Debiased Machine Learning for Treatment and Structural
Parameters.” *The Econometrics Journal* 21 (1): C1–68.
<https://doi.org/10.1111/ectj.12097>.

Czado, Claudia. 2019. *Analyzing Dependent Data with Vine Copulas: A
Practical Guide with R*. Vol. 222. Lecture Notes in Statistics.
Springer. <https://doi.org/10.1007/978-3-030-13785-4>.

Daley, D. J., and D. Vere-Jones. 2003. *An Introduction to the Theory of
Point Processes, Volume I: Elementary Theory and Methods*. 2nd ed.
Springer.

Diggle, Peter J. 2003. *Statistical Analysis of Spatial Point Patterns*.
2nd ed. Edward Arnold.

Dong, Yingying, and Arthur Lewbel. 2015. “A Simple Estimator for Binary
Choice Models with Endogenous Regressors.” *Econometric Reviews* 34
(1–2): 82–105. <https://doi.org/10.1080/07474938.2014.944470>.

Donoho, David L., and Iain M. Johnstone. 1994. “Ideal Spatial Adaptation
by Wavelet Shrinkage.” *Biometrika* 81 (3): 425–55.
<https://doi.org/10.1093/biomet/81.3.425>.

Doob, Anthony N. 2020. *Affidavit of Anthony N. Doob*. Federal Court of
Canada, File T-539-20, Application Record Vol. 3 of 5, pp. 778–795.

Engle, Robert F. 2002. “Dynamic Conditional Correlation: A Simple Class
of Multivariate Generalized Autoregressive Conditional
Heteroskedasticity Models.” *Journal of Business & Economic Statistics*
20 (3): 339–50. <https://doi.org/10.1198/073500102288618487>.

Fan, Jianqing. 1991. “On the Optimal Rates of Convergence for
Nonparametric Deconvolution Problems.” *The Annals of Statistics* 19
(3): 1257–72. <https://doi.org/10.1214/aos/1176348248>.

Fan, Rong-En, Pai-Hsuen Chen, and Chih-Jen Lin. 2005. “Working Set
Selection Using Second Order Information for Training Support Vector
Machines.” *Journal of Machine Learning Research* 6: 1889–918.

Fisher, Ronald A. 1921. “On the ‘Probable Error’ of a Coefficient of
Correlation Deduced from a Small Sample.” *Metron* 1: 3–32.

Friedman, Jerome H. 2001. “Greedy Function Approximation: A Gradient
Boosting Machine.” *The Annals of Statistics* 29 (5): 1189–232.
<https://doi.org/10.1214/aos/1013203451>.

Geyer, Charles J. 1992. “Practical Markov Chain Monte Carlo.”
*Statistical Science* 7 (4): 473–83.
<https://doi.org/10.1214/ss/1177011137>.

Gibbons, Jean Dickinson, and Subhabrata Chakraborti. 2011.
*Nonparametric Statistical Inference*. 5th ed. Statistics: Textbooks and
Monographs. Chapman & Hall/CRC.

Goffman, Erving. 1961. *Asylums: Essays on the Social Situation of
Mental Patients and Other Inmates*. Anchor Books.

Grassberger, Peter, and Itamar Procaccia. 1983. “Measuring the
Strangeness of Strange Attractors.” *Physica D: Nonlinear Phenomena* 9
(1–2): 189–208. <https://doi.org/10.1016/0167-2789(83)90298-1>.

Greenacre, Michael J. 1984. *Theory and Applications of Correspondence
Analysis*. Academic Press.

Gretton, Arthur, Karsten M. Borgwardt, Malte J. Rasch, Bernhard
Schölkopf, and Alexander Smola. 2012. “A Kernel Two-Sample Test.”
*Journal of Machine Learning Research* 13: 723–73.
<https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf>.

Han, Aaron K. 1987. “Non-Parametric Analysis of a Generalized Regression
Model: The Maximum Rank Correlation Estimator.” *Journal of
Econometrics* 35 (2–3): 303–16.
<https://doi.org/10.1016/0304-4076(87)90030-3>.

Han, Te Sun. 1978. “Nonnegative Entropy Measures of Multivariate
Symmetric Correlations.” *Information and Control* 36 (2): 133–56.
<https://doi.org/10.1016/S0019-9958(78)90275-9>.

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. *The
Elements of Statistical Learning: Data Mining, Inference, and
Prediction*. 2nd ed. Springer Series in Statistics. Springer.
<https://doi.org/10.1007/978-0-387-84858-7>.

Hedges, Larry V., Elizabeth Tipton, and Matthew C. Johnson. 2010.
“Robust Variance Estimation in Meta-Regression with Dependent Effect
Size Estimates.” *Research Synthesis Methods* 1 (1): 39–65.
<https://doi.org/10.1002/jrsm.5>.

Higuchi, T. 1988. “Approach to an Irregular Time Series on the Basis of
the Fractal Theory.” *Physica D: Nonlinear Phenomena* 31 (2): 277–83.
<https://doi.org/10.1016/0167-2789(88)90081-4>.

Horowitz, Joel L. 1992. “A Smoothed Maximum Score Estimator for the
Binary Response Model.” *Econometrica* 60 (3): 505–31.
<https://doi.org/10.2307/2951582>.

Horowitz, Joel L. 1996. “Semiparametric Estimation of a Regression Model
with an Unknown Transformation of the Dependent Variable.”
*Econometrica* 64 (1): 103–37. <https://doi.org/10.2307/2171926>.

Horowitz, Joel L. 2009. *Semiparametric and Nonparametric Methods in
Econometrics*. Springer Series in Statistics. Springer.
<https://doi.org/10.1007/978-0-387-92870-8>.

Horowitz, Joel L., and Marianthi Markatou. 1996. “Semiparametric
Estimation of Regression Models for Panel Data.” *The Review of Economic
Studies* 63 (1): 145–68. <https://doi.org/10.2307/2298119>.

Hosking, J. R. M. 1980. “The Multivariate Portmanteau Statistic.”
*Journal of the American Statistical Association* 75 (371): 602–8.
<https://doi.org/10.1080/01621459.1980.10477520>.

Ichimura, Hidehiko. 1993. “Semiparametric Least Squares (SLS) and
Weighted SLS Estimation of Single-Index Models.” *Journal of
Econometrics* 58 (1–2): 71–120.
<https://doi.org/10.1016/0304-4076(93)90114-K>.

Imai, Kosuke, Luke Keele, and Teppei Yamamoto. 2010. “Identification,
Inference and Sensitivity Analysis for Causal Mediation Effects.”
*Statistical Science* 25 (1): 51–71.
<https://doi.org/10.1214/10-STS321>.

Imbens, Guido W. 2003. “Sensitivity to Exogeneity Assumptions in Program
Evaluation.” *American Economic Review* 93 (2): 126–32.
<https://doi.org/10.1257/000282803321946921>.

Jacquez, Geoffrey M. 1996. “A k Nearest Neighbour Test for Space-Time
Interaction.” *Statistics in Medicine* 15 (18): 1935–49.
[https://doi.org/10.1002/(SICI)1097-0258(19960930)15:18\<1935::AID-SIM406\>3.0.CO;2-I](https://doi.org/10.1002/(SICI)1097-0258(19960930)15:18<1935::AID-SIM406>3.0.CO;2-I).

Klein, Roger W., and Richard H. Spady. 1993. “An Efficient
Semiparametric Estimator for Binary Response Models.” *Econometrica* 61
(2): 387–421. <https://doi.org/10.2307/2951556>.

Kooreman, Peter, and Bertrand Melenberg. 1989. *Maximum Score Estimation
in the Ordered Response Model*. Discussion Paper Nos. 1989-48. Tilburg
University, Center for Economic Research.

Kosorok, Michael R. 2008. *Introduction to Empirical Processes and
Semiparametric Inference*. Springer Series in Statistics. Springer.
<https://doi.org/10.1007/978-0-387-74978-5>.

Laan, Mark J. van der, Eric C. Polley, and Alan E. Hubbard. 2007. “Super
Learner.” *Statistical Applications in Genetics and Molecular Biology* 6
(1).

Lewbel, Arthur. 2000. “Semiparametric Qualitative Response Model
Estimation with Unknown Heteroscedasticity or Instrumental Variables.”
*Journal of Econometrics* 97 (1): 145–77.
<https://doi.org/10.1016/S0304-4076(00)00015-4>.

Lumley, Thomas. 2010. *Complex Surveys: A Guide to Analysis Using R*.
Wiley. <https://doi.org/10.1002/9780470580066>.

Mahdi, Esam. 2020. *Portes: An R Package for Portmanteau Tests in Time
Series Models*. <https://arxiv.org/abs/2005.00931>.

Manski, Charles F. 1985. “Semiparametric Analysis of Discrete Response:
Asymptotic Properties of the Maximum Score Estimator.” *Journal of
Econometrics* 27 (3): 313–33.
<https://doi.org/10.1016/0304-4076(85)90009-0>.

Manski, Charles F. 1987. “Semiparametric Analysis of Random Effects
Linear Models from Binary Panel Data.” *Econometrica* 55 (2): 357–62.
<https://doi.org/10.2307/1913020>.

Marple, S. Lawrence. 1987. *Digital Spectral Analysis with
Applications*. Prentice-Hall.

Matthews, Brian W. 1975. “Comparison of the Predicted and Observed
Secondary Structure of T4 Phage Lysozyme.” *Biochimica Et Biophysica
Acta (BBA) — Protein Structure* 405 (2): 442–51.
<https://doi.org/10.1016/0005-2795(75)90109-9>.

Melenberg, Bertrand, and Arthur van Soest. 1996. “Measuring the Costs of
Children: Parametric and Semiparametric Estimators.” *Statistica
Neerlandica* 50 (1): 171–92.
<https://doi.org/10.1111/j.1467-9574.1996.tb01486.x>.

Moran, P. A. P. 1950. “Notes on Continuous Stochastic Phenomena.”
*Biometrika* 37 (1/2): 17–23. <https://doi.org/10.2307/2332142>.

Nenadic, Oleg, and Michael Greenacre. 2007. “Correspondence Analysis in
R, with Two- and Three-Dimensional Graphics: The Ca Package.” *Journal
of Statistical Software* 20 (3): 1–13.
<https://doi.org/10.18637/jss.v020.i03>.

Newey, Whitney K., and Thomas M. Stoker. 1993. “Efficiency of Weighted
Average Derivative Estimators and Index Models.” *Econometrica* 61 (5):
1199–223. <https://doi.org/10.2307/2951498>.

Nunnally, Jum C., and Ira H. Bernstein. 1994. *Psychometric Theory*. 3rd
ed. McGraw-Hill.

Pearson, Karl. 1897. “Mathematical Contributions to the Theory of
Evolution.—on a Form of Spurious Correlation Which May Arise When
Indices Are Used in the Measurement of Organs.” *Proceedings of the
Royal Society of London* 60: 489–98.
<https://doi.org/10.1098/rspl.1896.0076>.

Pedroni, Peter. 1999. “Critical Values for Cointegration Tests in
Heterogeneous Panels with Multiple Regressors.” *Oxford Bulletin of
Economics and Statistics* 61 (S1): 653–70.
<https://doi.org/10.1111/1468-0084.61.s1.14>.

Peng, C.-K., S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and A.
L. Goldberger. 1994. “Mosaic Organization of DNA Nucleotides.” *Physical
Review E* 49 (2): 1685–89. <https://doi.org/10.1103/PhysRevE.49.1685>.

Percival, Donald B., and Andrew T. Walden. 2000. *Wavelet Methods for
Time Series Analysis*. Cambridge Series in Statistical and Probabilistic
Mathematics. Cambridge University Press.

Pincus, Steven M. 1991. “Approximate Entropy as a Measure of System
Complexity.” *Proceedings of the National Academy of Sciences* 88 (6):
2297–301. <https://doi.org/10.1073/pnas.88.6.2297>.

Powell, James L., James H. Stock, and Thomas M. Stoker. 1989.
“Semiparametric Estimation of Index Coefficients.” *Econometrica* 57
(6): 1403–30. <https://doi.org/10.2307/1913713>.

Ramdas, Aaditya, Nicolás García Trillos, and Marco Cuturi. 2017. “On
Wasserstein Two-Sample Testing and Related Families of Nonparametric
Tests.” *Entropy* 19 (2): 47. <https://doi.org/10.3390/e19020047>.

Ramsay, James O., and Bernard W. Silverman. 2005. *Functional Data
Analysis*. 2nd ed. Springer. <https://doi.org/10.1007/b98888>.

Rangayyan, Rangaraj M., and Sridhar Krishnan. 2024. *Biomedical Signal
Analysis*. Third. IEEE Press Series in Biomedical Engineering. IEEE
Press / Wiley.

Richman, Joshua S., and J. Randall Moorman. 2000. “Physiological
Time-Series Analysis Using Approximate Entropy and Sample Entropy.”
*American Journal of Physiology-Heart and Circulatory Physiology* 278
(6): H2039–49. <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>.

Ripley, Brian D. 1977. “Modelling Spatial Patterns.” *Journal of the
Royal Statistical Society, Series B* 39 (2): 172–212.
<https://doi.org/10.1111/j.2517-6161.1977.tb01615.x>.

Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. 1993. “A
Practical Method for Calculating Largest Lyapunov Exponents from Small
Data Sets.” *Physica D: Nonlinear Phenomena* 65 (1–2): 117–34.
<https://doi.org/10.1016/0167-2789(93)90009-P>.

Ruhela, Vansh Singh. 2026. “The MRM Framework: A Multi-Source
Statistical Foundation for Canadian Carceral, Police, and Oversight
Data, Implemented as MRM Modules in MORIE.” Unpublished manuscript.

Samejima, Fumiko. 1969. *Estimation of Latent Ability Using a Response
Pattern of Graded Scores*. Psychometrika Monograph Supplement, No. 17.
Psychometric Society.

Schabenberger, Oliver, and Carol A. Gotway. 2005. *Statistical Methods
for Spatial Data Analysis*. Chapman; Hall/CRC.

Shealy, Robin, and William Stout. 1993. “A Model-Based Standardization
Approach That Separates True Bias/DIF from Group Ability Differences and
Detects Test Bias/DTF as Well as Item Bias/DIF.” *Psychometrika* 58 (2):
159–94. <https://doi.org/10.1007/BF02294572>.

Sherman, Robert P. 1993. “The Limiting Distribution of the Maximum Rank
Correlation Estimator.” *Econometrica* 61 (1): 123–37.
<https://doi.org/10.2307/2951780>.

Short, M. B., M. R. D’Orsogna, V. B. Pasour, et al. 2008. “A Statistical
Model of Criminal Behavior.” *Mathematical Models and Methods in Applied
Sciences* 18 (Suppl.): 1249–67.

Spielman, Richard S., Ralph E. McGinnis, and Warren J. Ewens. 1993.
“Transmission Test for Linkage Disequilibrium: The Insulin Gene Region
and Insulin-Dependent Diabetes Mellitus (IDDM).” *American Journal of
Human Genetics* 52 (3): 506–16.

Sprott, Jane B., and Anthony N. Doob. 2020a. *Is There Clear Evidence
That COVID-19 Was the Cause of Problems with the Operation of CSC’s
Structured Intervention Units?* Centre for Criminology; Sociolegal
Studies, University of Toronto.

Sprott, Jane B., and Anthony N. Doob. 2020b. *Understanding the
Operation of Correctional Service Canada’s Structured Intervention
Units: Some Preliminary Findings*. Centre for Criminology; Sociolegal
Studies, University of Toronto.

Sprott, Jane B., and Anthony N. Doob. 2021. *Solitary Confinement,
Torture, and Canada’s Structured Intervention Units*. Centre for
Criminology; Sociolegal Studies, University of Toronto.

Sprott, Jane B., Anthony N. Doob, and Adelina Iftene. 2021. *Do
Independent External Decision Makers Ensure That “an Inmate’s
Confinement in a Structured Intervention Unit Is to End as Soon as
Possible”? \[Corrections and Conditional Release Act, Section 33\]*.
Schulich School of Law, Dalhousie University.

Task Force of the European Society of Cardiology and the North American
Society of Pacing and Electrophysiology. 1996. “Heart Rate Variability:
Standards of Measurement, Physiological Interpretation, and Clinical
Use.” *Circulation* 93 (5): 1043–65.
<https://doi.org/10.1161/01.CIR.93.5.1043>.

Tsay, Ruey S. 2010. *Analysis of Financial Time Series*. 3rd ed. Wiley.
<https://doi.org/10.1002/9780470644560>.

United Nations General Assembly. 2015. *United Nations Standard Minimum
Rules for the Treatment of Prisoners (the Nelson Mandela Rules)*.
A/Res/70/175.

VanderWeele, Tyler J., and Peng Ding. 2017. “Sensitivity Analysis in
Observational Research: Introducing the E-Value.” *Annals of Internal
Medicine* 167 (4): 268–74.

VanRaden, P. M. 2008. “Efficient Methods to Compute Genomic
Predictions.” *Journal of Dairy Science* 91 (11): 4414–23.
<https://doi.org/10.3168/jds.2007-0980>.

Villani, Cédric. 2009. *Optimal Transport: Old and New*. Vol. 338.
Grundlehren Der Mathematischen Wissenschaften. Springer.
<https://doi.org/10.1007/978-3-540-71050-9>.

Wallace, Marnie, John Turner, Anthony Matarazzo, and Colin Babyak. 2009.
*Measuring Crime in Canada: Introducing the Crime Severity Index and
Improvements to the Uniform Crime Reporting Survey*. 85-004-X.
Statistics Canada.

Watanabe, Satosi. 1960. “Information Theoretical Analysis of
Multivariate Correlation.” *IBM Journal of Research and Development* 4
(1): 66–82. <https://doi.org/10.1147/rd.41.0066>.

Welch, Peter D. 1967. “The Use of Fast Fourier Transform for the
Estimation of Power Spectra: A Method Based on Time Averaging over
Short, Modified Periodograms.” *IEEE Transactions on Audio and
Electroacoustics* 15 (2): 70–73.
<https://doi.org/10.1109/TAU.1967.1161901>.

Widrow, Bernard, and Samuel D. Stearns. 1985. *Adaptive Signal
Processing*. Prentice-Hall.

Zandieh, Amir, Majid Daliri, Milad Hadian, and Vahab Mirrokni. 2026.
“TurboQuant: Online Vector Quantization with Near-Optimal Distortion
Rate.” *International Conference on Learning Representations (ICLR)*.
<https://doi.org/10.48550/arXiv.2504.19874>.
