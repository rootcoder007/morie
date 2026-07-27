# Acknowledgments

MORIE (Multi-domain Open Research and Inferential Estimation) is developed by Vansh Singh Ruhela via the Collaborative Specialization in Addiction Studies (CoPAS) between Dalla Lana School of Public Health (DLSPH) and the Centre for Criminology and Sociolegal Studies (CrimSL) at the University of Toronto School of Graduate Studies (UTSGS).

## AI Development Partners

**Anthropic / Claude** — Claude (Opus, Sonnet) serves as the AI co-architect of MORIE, contributing to code generation, statistical function design, testing infrastructure, documentation, and architectural decisions across 2000+ function implementations. Claude Code is integral to the MORIE development workflow.

**Google / Gemma** — The Gemma model family (Gemma 3, Gemma 4) powers Perseus, MORIE's resident AI agent. Perseus (`perseus:e2b`) is a custom-tuned Gemma 4 model (7.2GB, Q4_K_M) with domain-specific expertise in causal inference, scientific experimentation, and statistical computing. Google's open model weights enable fully local, private AI inference.

## Frameworks and Tools

**Jeroen Ooms / r-universe** — [Jeroen Ooms](https://github.com/jeroen) (rOpenSci, University of California, Berkeley) maintains the [r-universe](https://r-universe.dev) infrastructure that builds and serves nightly Linux + macOS + Windows binaries of the `morie` R package at [rootcoder007.r-universe.dev](https://rootcoder007.r-universe.dev). r-universe's CRAN-like service is what makes Linux-binary R installation tractable for downstream users without requiring source compilation.

**DoubleML team** — [DoubleML](https://github.com/DoubleML/doubleml-for-py) (Bach, Chernozhukov, Klaassen, Kurz, Spindler) is MORIE's canonical double-machine-learning back-end. The Python and R packages are released under BSD-3-Clause; MORIE wraps DoubleML for its IRM, PLR, and PLIV estimators, with Python and R idiomatic APIs around them. See Bach et al. (2022, *Journal of Statistical Software*, v108i03).

**Andrej Karpathy / autoresearch** — The [autoresearch](https://github.com/karpathy/autoresearch) framework provided the foundation for MORIE's autonomous LLM pretraining experiments, including the 50.3M parameter model trained on Apple Silicon (M2) with MPS acceleration.

**TurboQuant** (Ankush Agarwal et al., ICLR 2026) — The TurboQuant algorithm (arxiv.org/abs/2504.19874) is implemented in `morie.quant` for data-oblivious, unbiased KV-cache compression. PolarQuant + QJL + Lloyd-Max codebooks achieve 4-bit compression at 0.995 cosine similarity.

**Ollama** — [Ollama](https://ollama.com) provides the local LLM serving infrastructure that powers Perseus on both macOS and Raspberry Pi. Ollama's model management, quantization support, and simple API make local AI inference accessible across platforms.

**OllamaFreeAPI** — Community-maintained free API providing access to 16+ LLM models without API keys, serving as MORIE's fallback provider when local models are unavailable.

## Open Source Dependencies

MORIE builds on the work of many open-source projects, including but not limited to: NumPy, SciPy, pandas, scikit-learn, Textual, httpx, Sphinx, and the broader Python/R scientific computing ecosystem.

## Methods and their authors

morie's general-purpose modules implement published methods. The
implementations are written against the primary sources, and each function's
docstring cites the specific chapter, section or equation, so a reader can
check the code against the same page the author wrote.

| Method | Source |
|---|---|
| Biomedical signal analysis (filtering, spectral, waveform complexity) | Rangaraj M. Rangayyan & Sridhar Krishnan, *Biomedical Signal Analysis*, 3rd ed. (IEEE Press / Wiley, 2024) |
| Higuchi fractal dimension | T. Higuchi, *Physica D* 31:277–283 (1988) |
| Correlation dimension D₂ | P. Grassberger & I. Procaccia, *Physica D* 9:189–208 (1983) |
| Detrended fluctuation analysis | C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley & A. L. Goldberger, *Phys. Rev. E* 49:1685–1689 (1994) |
| Approximate entropy | S. M. Pincus, *PNAS* 88:2297–2301 (1991) |
| Sample entropy | J. S. Richman & J. R. Moorman, *Am. J. Physiol.* 278:H2039–H2049 (2000) |
| Graded response model (polytomous IRT) | F. Samejima, *Psychometrika Monograph Supplement* No. 17 (1969) |
| Genomic relationship matrices | P. M. VanRaden, *J. Dairy Sci.* 91:4414–4423 (2008) |
| LMS adaptive noise cancelling | Bernard Widrow & Samuel D. Stearns, *Adaptive Signal Processing* (Prentice-Hall, 1985) |
| Autoregressive modelling (Burg's recursion) | John Parker Burg, *Maximum Entropy Spectral Analysis* (PhD thesis, Stanford University, 1975); S. Lawrence Marple, *Digital Spectral Analysis* (Prentice-Hall, 1987) |
| Averaged-periodogram power spectral density | Peter D. Welch, *IEEE Trans. Audio Electroacoust.* 15:70–73 (1967) |
| Heart-rate variability (SDNN, RMSSD, pNN50) | Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, *Circulation* 93:1043–1065 (1996) |
| Wavelet shrinkage denoising (universal threshold) | David L. Donoho & Iain M. Johnstone, *Biometrika* 81:425–455 (1994) |
| Discrete wavelet transform for time series | Donald B. Percival & Andrew T. Walden, *Wavelet Methods for Time Series Analysis* (Cambridge University Press, 2000) |
| Largest Lyapunov exponent | Michael T. Rosenstein, James J. Collins & Carlo J. De Luca, *Physica D* 65:117–134 (1993) |
| Global spatial autocorrelation (Moran's I) | P. A. P. Moran, *Biometrika* 37:17–23 (1950); Oliver Schabenberger & Carol A. Gotway, *Statistical Methods for Spatial Data Analysis* (Chapman and Hall/CRC, 2005) |
| Correlation variance-stabilising transform | Ronald A. Fisher, *Metron* 1:3–32 (1921) |
| Total correlation (multi-information) | Satosi Watanabe, *IBM Journal of Research and Development* 4:66–82 (1960) |
| Dual total correlation | Te Sun Han, *Information and Control* 36:133–156 (1978) |
| Corrected item-total correlation | Jum C. Nunnally & Ira H. Bernstein, *Psychometric Theory*, 3rd ed. (McGraw-Hill, 1994) |
| Variance inflation for dependent effect sizes | Larry V. Hedges, Elizabeth Tipton & Matthew C. Johnson, *Research Synthesis Methods* 1:39–65 (2010) |
| Transmission disequilibrium test | Richard S. Spielman, Ralph E. McGinnis & Warren J. Ewens, *American Journal of Human Genetics* 52:506–516 (1993) |
| MCMC effective sample size (initial positive sequence) | Charles J. Geyer, *Statistical Science* 7:473–483 (1992) |
| Functional data correlation | James O. Ramsay & Bernard W. Silverman, *Functional Data Analysis*, 2nd ed. (Springer, 2005) |
| Double machine learning | V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey & J. Robins, *Econometrics Journal* 21:C1–C68 (2018) |
| Anderson–Darling and Lilliefors goodness-of-fit (critical-value tables) | Jean Dickinson Gibbons & Subhabrata Chakraborti, *Nonparametric Statistical Inference*, 5th ed. (Chapman & Hall/CRC, 2011) |
| Constant conditional correlation MGARCH | Tim Bollerslev, *Review of Economics and Statistics* 72:498–505 (1990) |
| Dynamic conditional correlation MGARCH | Robert F. Engle, *Journal of Business & Economic Statistics* 20:339–350 (2002) |
| Random forests | Leo Breiman, *Machine Learning* 45:5–32 (2001); Trevor Hastie, Robert Tibshirani & Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed. (Springer, 2009), Algorithm 15.1 |
| Gradient boosting | Jerome H. Friedman, *Annals of Statistics* 29:1189–1232 (2001); Hastie, Tibshirani & Friedman (2009), Algorithm 10.3 and eq. (10.41) |
| Regularised tree-boosting objective | Tianqi Chen & Carlos Guestrin, *KDD '16*, 785–794 (2016) |
| Support vector machines (SMO decomposition) | Chih-Chung Chang & Chih-Jen Lin, *ACM TIST* 2(3):27 (2011); Rong-En Fan, Pai-Hsuen Chen & Chih-Jen Lin, *JMLR* 6:1889–1918 (2005) |
| Causal mediation sensitivity to an unobserved confounder | Kosuke Imai, Luke Keele & Teppei Yamamoto, *Statistical Science* 25:51–71 (2010); Guido W. Imbens, *American Economic Review* 93:126–132 (2003) |
| Compositional data, closure and subcompositional coherence | John Aitchison, *The Statistical Analysis of Compositional Data* (Chapman & Hall, 1986); Karl Pearson, *Proceedings of the Royal Society of London* 60:489–498 (1897) |
| Matthews correlation coefficient | Brian W. Matthews, *Biochimica et Biophysica Acta* 405:442–451 (1975) |
| Variance inflation factor and collinearity diagnostics | David A. Belsley, Edwin Kuh & Roy E. Welsch, *Regression Diagnostics* (Wiley, 1980) |
| Complete spatial randomness, nearest-neighbour distances | Peter J. Diggle, *Statistical Analysis of Spatial Point Patterns*, 2nd ed. (Edward Arnold, 2003); Schabenberger & Gotway (2005), §§3.3–3.4 |

Where a secondary source and the primary disagree, morie follows the primary
and records the divergence in the function's docstring, so the choice is
auditable rather than inherited. The genomic relationship matrices are the
worked case: a widely used secondary text renumbers VanRaden's three methods,
and morie's method aliases follow VanRaden's own numbering.

## Funding

This work is conducted with zero external funding, using exclusively free-tier and open-source tools.
