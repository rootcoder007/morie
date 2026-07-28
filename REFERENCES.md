# References

Bibliography for `paper.md`, rendered from `paper.bib`.

`paper.md` is a JOSS submission source: its `# References` heading is
deliberately empty because Pandoc's citeproc fills it during the JOSS
build. GitHub renders markdown without citeproc, so on GitHub that
heading looks blank and the inline citations show as raw `[-@key]`
tokens. This file exists so the bibliography is readable in the
repository itself.

**Generated, do not hand-edit.** Regenerate with:

```sh
pandoc paper.md --citeproc --bibliography=paper.bib -t gfm -o - \
  | sed -n '/^# References/,$p'
```

94 entries, last regenerated 2026-07-28.

---

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
