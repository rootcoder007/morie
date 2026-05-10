# moirais 0.1.0-4

* Initial CRAN submission.
* 99 exported functions across causal inference (ATE/ATT/ATC/GATE/CATE/LATE,
  AIPW, G-computation, IRM via DoubleML, IPW, AIPW, Rosenbaum bounds,
  E-value), survey sampling (stratified/cluster/PPS/bootstrap/jackknife,
  calibration weights, design effects), psychometric and effect-size
  helpers (Cohen's d, Hedges' g, η², ω², Cramér's V, Kendall's τ,
  Spearman's ρ), classical statistical tests (one-/two-sample/paired
  t, Wilcoxon, Mann-Whitney, Kruskal-Wallis, Levene, Shapiro-Wilk, χ²,
  Fisher exact), confidence intervals (risk-difference, risk-ratio,
  odds-ratio, proportion), power and sample-size (`power_t_test`,
  `power_prop_test`, `sample_size_logistic`), signal-processing primitives
  (Butterworth filters, Higuchi fractal dimension, Hurst exponent),
  dataset profiling, OTIS correctional-data analysis, and the MRM
  (McNamara-Ruhela-Medina) framework.
* Python parity: this package is the R sibling of the Python `moirais`
  package on PyPI. Both expose the same conceptual public API; each uses
  its native language's idioms and ML ecosystem (R: mlr3 + DoubleML;
  Python: scikit-learn + DoubleML).
* `estimate_irm()` is a thin R wrapper around `DoubleML::DoubleMLIRM` from
  the CRAN `DoubleML` package; `DoubleML`, `mlr3`, and `mlr3learners` are in
  `Suggests` and the function gates them with `requireNamespace()`.
* CITATION includes both the software DOI (`10.5281/zenodo.20111233`) and
  the companion paper DOI (`10.5281/zenodo.20096350`).
