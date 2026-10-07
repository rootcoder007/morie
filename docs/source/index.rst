.. meta::
   :description: MORIE is a multi-domain scientific computing toolkit (Python and R) for observational causal inference, survey methods, spatial statistics, and criminal-justice analytics, and the home of the MRM (Multilevel Reconciliation Methodology) framework.
   :keywords: MORIE, MRM framework, causal inference, observational inference, double machine learning, criminology, survey statistics, R package, Python package
   :google-site-verification: 8ORzeHrfoPIOrujfV-km-kXCi9HD8dvL5GYH-GqwjYI

MORIE 森
========

**Multi-domain Open Research and Inferential Estimation.**

.. image:: https://img.shields.io/badge/license-AGPL--3.0--or--later-a42e2b.svg
   :alt: License: AGPL-3.0-or-later

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/R-4.3%2B-276DC3.svg
   :alt: R 4.3+

.. image:: https://img.shields.io/pypi/v/morie.svg
   :target: https://pypi.org/project/morie/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/r--universe-rootcoder007-276DC3
   :target: https://rootcoder007.r-universe.dev/rmorie
   :alt: r-universe

A dual-language (Python + R) multi-domain scientific computing toolkit for
observational inference, with sociolegal, signal-processing, cryptographic,
spatial-statistics, statistical-physics, and psychometrics modules. Hosts
the MRM (Multilevel Reconciliation Methodology) framework as a primary application
for Canadian carceral, police, and oversight data analysis.

----

Quick start
-----------

Pick any one channel — each installs the current ``morie`` release:

.. code-block:: bash

   # 1. One-line installer (Linux / macOS / WSL) — detects pip + R, installs both
   curl -fsSL https://rootcoder007.github.io/morie/install.sh | bash

   # 2. PyPI (any platform with Python ≥3.10)
   pip install morie                  # 71 catalogued datasets, 15,222 morie.fn callables
   pip install "morie[interactive]"   # + Terminal IDE (TUI)
   morie interactive install          # the REPL, exec, agent and TUI modules, once per user
   #    Debian / Ubuntu / Raspberry Pi OS refuse system-wide pip: use a venv
   python3 -m venv ~/.venvs/morie && source ~/.venvs/morie/bin/activate && pip install -U morie

   # 3. Homebrew (macOS / Linuxbrew)
   brew tap rootcoder007/morie
   brew install morie
   morie interactive install

   # 4. Docker (zero local dependencies)
   docker run --rm ghcr.io/rootcoder007/morie:latest morie --help

   # 5. R package: rmorie (r-universe; its companions rmoriebricklayer
   #    and rmoriedata are on CRAN)
   install.packages("rmorie", repos = c("https://rootcoder007.r-universe.dev",
                                        "https://cloud.r-project.org"))

.. note::

   **Heads-up for Raspberry Pi OS / Debian Trixie users:** the system
   ``/usr/bin/python3`` is python 3.13.5, which segfaults on import for
   several scientific wheels (a Debian-packaging bug, not a morie bug).
   Work around it with ``uv``:

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh
      uv python install 3.12
      uv venv ~/.venvs/morie --python 3.12
      uv pip install --python ~/.venvs/morie/bin/python morie
      ~/.venvs/morie/bin/morie --version

Run your first analysis in seconds:

.. code-block:: bash

   # Launch the Terminal IDE (multi-pane IDE)
   morie tui

   # Self-diagnostics — checks LLM providers, datasets, R, Docker
   morie doctor

   # List all 71 catalogued datasets
   morie list-datasets

   # List all 23 analysis modules
   morie list-modules

   # Run a single module against built-in data
   morie run-module power-design --output-dir /tmp/morie-outputs

   # Run the full pipeline (with enlighten progress bars)
   morie pipeline --all -y

   # Chat with the local model (no API key needed)
   morie chat

From R:

.. code-block:: r

   library(rmorie)

   # Load a built-in dataset by key (CSV corpus from rmoriedata; no file
   # paths) and map the PUMF columns to the canonical analysis names
   cpads <- morie_canonicalize_cpads_data(morie_load_dataset("ocp21"))

   # List all built-in datasets
   morie_list_datasets()

   # Browse the dataset catalog
   morie_dataset_catalog()

   # Estimate an average treatment effect
   ate <- morie_estimate_ate(cpads, treatment = "cannabis_any_use",
                             outcome = "heavy_drinking_30d",
                             covariates = c("age_group", "gender"))

----

What MORIE does
-----------------

A unified Python + R interface across the following surfaces. See
:doc:`methods/index` for methodology details and :doc:`api/index`
for function reference.

**Causal estimators**
  ATE, ATT, ATC, GATE, CATE (T- / S-learner), LATE (2SLS / Wald),
  AIPW, IPW (Hájek), G-computation, propensity-score matching
  (1:1 NN, 5-strata subclass), Rosenbaum sensitivity bounds, E-value.

**Double machine learning**
  Partially linear regression (PLR), interactive regression model
  (IRM), partially linear IV (PLIV); cross-fitted with pluggable
  nuisance learners. Multi-SE comparison (pooled, cluster, multi-way)
  on the IRM-DML primary estimate. Propensity calibration (Platt /
  isotonic) on IPW / AIPW / SuperLearner-AIPW with Brier score.

**The MRM framework**
  Multilevel Reconciliation Methodology — a 10-estimator framework applied to OTIS
  / SIU / TPS data over a coordinated set of (treatment, outcome,
  covariates) designs. Per-row individual-level + aggregate (Poisson,
  NB GLM) modes. Mandela classifier (UN Mandela Rules 43 + 44) +
  provincial-vs-federal cross-comparison.

**Spatial statistics**
  Moran's :math:`I`, Geary's :math:`C`, Getis-Ord general :math:`G`,
  join count, LISA, Getis-Ord :math:`G_i^{*}`, local Geary, Ripley's
  K / L, geostatistical kriging (ordinary, universal, IDW,
  co-kriging), variogram fitting, GWR (basic, GW-PCA, ST-GWR),
  bivariate Moran, Moran sweep heatmap, DBSCAN / HDBSCAN, Kulldorff
  space-time scan.

**Hawkes self-exciting point processes**
  Markovian Mohler-Bertozzi-Brantingham fit (exponential kernel +
  constant baseline) plus the non-stationary, non-Markovian
  Kwan-Chen-Dunsmuir (2024) family — Gamma, Weibull, Lomax kernels
  with sinusoidal baselines. Eight (kernel × baseline) combinations
  ranked by AIC and time-rescaling Kolmogorov-Smirnov goodness-of-fit.

**Statistical physics of crime**
  Short-Brantingham reaction-diffusion PDE, Brockmann-Hufnagel-Geisel
  Lévy-flight tail (Hill estimator), Bettencourt urban-scaling
  exponent (HC3-OLS), D'Orsogna-Perc Lotka-Volterra predator-prey,
  SDB Turing-pattern demo, Helbing-Szolnoki inspection-game phase
  diagram, criminal-role co-occurrence networks.

**Survey-weighted inference**
  Horvitz-Thompson totals, Hájek means, ratio estimators, calibration
  weights (raking / IPF), complex-survey GLM, subpopulation estimates,
  stratified / cluster / PPS sampling, bootstrap + jackknife variance,
  design-effect computations, effective-sample-size diagnostics.

**Psychometrics**
  Cronbach's :math:`\alpha`, McDonald's :math:`\omega_t` /
  :math:`\omega_h`, KMO sampling adequacy, Bartlett's sphericity,
  parallel analysis, composite reliability, AVE, item-response-theory
  fits (1PL / 2PL / 3PL / GRM / PCM), differential item functioning
  (Mantel-Haenszel, logistic, generalised), measurement invariance,
  network psychometrics, Bayesian psychometrics. 250+ functions.

**Signal processing + cryptography**
  Spectral analysis, biomedical-signal helpers, homomorphic
  deconvolution, classical and modern crypto primitives
  (ChaCha20-Poly1305, etc.), TurboQuant vector quantization with
  near-optimal distortion (Zandieh et al. 2026 ICLR).

**Datasets**
  71 catalog keys (Canadian carceral, police, and oversight +
  epidemiological reference data). 70 download themselves on first use
  and are cached: open.canada.ca, data.ontario.ca, Statistics Canada,
  CIHI, the Health Infobase tables (with the data.rmorie.com copy as the
  fallback), the NAPS air-quality files, the Toronto Police feeds, the
  OTIS research environments from data.rmorie.com and the reviewed SIU
  corpus from ``rmoriedata`` on CRAN. One, the MAPQ workbook, is your own
  file, resolved through ``MORIE_DATA_DIR``; ``morie list-datasets`` shows
  the route of every key and the curated data.rmorie.com tables once a key
  from https://rmorie.com/access is stored with ``morie login --token``.
  Small synthetic
  samples of the core tables ship in the wheel for tests and tutorials.
  Auto dataset-profiling for arbitrary tabular input
  (``morie.dataset.profile_dataset``).

**Function namespace ``morie.fn``**
  15,222 individual callables indexed by a registry, exposing
  short stable names for every estimator, every kernel,
  every weight matrix, every test. Use ``morie.fn.cheatsheet(name)``
  for a per-function help card.

**Federal SIU + Doob T-539-20 replication**
  Mandela classifier (Rules 43 + 44) with χ² verification, Sprott /
  Doob (± Iftene) IEDM analyses, full replication of Doob's CCRSO 2018
  Tables 1--3 and the imprisonment-vs-crime decoupling Pettitt
  change-point test. See :doc:`methods/sprott_doob`,
  :doc:`methods/doob_trends`, :doc:`methods/siuiap`.

**Toronto Police Service surface**
  ``morie.tps_*`` modules: incident I/O, CSI, neighbourhood
  spatial / temporal analyses, Hawkes (basic + advanced),
  statistical physics, Hohl-style choropleths and proportional-symbol
  district maps. (See CITATION.cff for the companion Hawkes paper.)

**LLM + assistant**
  Providers in order: a local Ollama (private, tried first) → your own
  Gemini / OpenAI-compatible / OpenAI keys → the hosted MORIE tier at
  ``llm.rmorie.com`` as a last resort (key on request at rmorie.com/access,
  stored with ``morie login --token``; rate-limited, nothing stored) → a
  keyword-matched local fallback that needs no network.
  See :doc:`hosted`. Vendored TurboQuant KV-cache compression. Polyglot
  REPL bridges variables across Python ↔ R ↔ shell ↔ 12 other
  languages.

**Carbon-aware computing**
  Built-in pure-Python emissions tracker
  (``morie.emissions``) with 213-country IEA carbon-intensity
  data, per-module and pipeline-wide CO₂ accounting. CodeCarbon
  fallback on Python ≤ 3.14.

----

Key design principles
---------------------

*Lean terminal IDE.*
  Rich terminal output — progress bars, formatted ASCII tables, color-coded
  diagnostics. Run entire pipelines from a single ``morie`` command.

*Python + R parity.*
  Every statistical estimator is implemented in both languages with matching
  ``morie_*`` names and arguments, and the two arms are checked against each
  other in the test suite. Neither arm depends on NumPy, pandas or an
  external estimator package: the numerics run on the family's own cores.

*Automated documentation.*
  Python API docs via Sphinx autodoc. R API docs via Roxygen2 → ``.. r:function::``
  (no manual writing). Run ``devtools::document()`` to regenerate.

*Data governance built-in.*
  Raw microdata is never committed: the real files are fetched from their
  source portals and cached locally. Synthetic data
  (``morie_generate_synthetic_data()``) is labeled synthetic in all
  outputs, and ``morie verify`` validates module outputs after a run.

*Statistically rigorous.*
  Target estimand is always an explicit parameter (ATE vs ATT vs CATE — never
  implicit). Overlap/positivity violations raise explicit warnings. Cross-fitting
  prevents data leakage. Convergence diagnostics are built into MCMC outputs.

----

Background
----------

MORIE is a multi-domain scientific computing toolkit for
observational inference. It sits between one-off research scripts
and heavy enterprise analytics platforms, and is aimed at
researchers who need:

- A unified Python + R surface across the same estimators (no
  language-choice tax).
- Causal estimators (ATE / ATT / ATC / GATE / CATE / LATE, AIPW,
  G-computation, DML--PLR, DML--IRM, propensity-score matching,
  E-value and Rosenbaum-bound sensitivity) with explicit estimands.
- Survey-weighted inference (Horvitz-Thompson, Hájek, raking,
  cluster + stratified design) on top of the same DataFrame as the
  causal layer.
- Spatial statistics (Moran's :math:`I`, LISA, Getis-Ord
  :math:`G^{*}`, DBSCAN, Kulldorff space-time scan), Hawkes
  self-exciting point processes (Markovian and non-Markovian), and
  the statistical-physics-of-crime models (Short-Brantingham
  reaction-diffusion, Lévy-flight tail, Bettencourt urban scaling,
  Lotka-Volterra) — applied as first-class methods on the
  Toronto Police Service open-data feeds.
- Reproducible pipelines that run unattended in CI / CD — outputs
  carry provenance manifests; synthetic data is labelled as such.
- The MRM (Multilevel Reconciliation Methodology) framework as a primary
  application for Canadian carceral, police, and oversight data
  (Ontario OTIS, federal SIU, TPS).

The package catalogues 71 datasets (Canadian carceral, police, and
oversight + epidemiological reference data): 70 download themselves from
their portal, from ``rmoriedata`` or from data.rmorie.com on first use,
one (the MAPQ workbook) is your own file placed under ``MORIE_DATA_DIR``;
the core tables also ship as synthetic samples.

MORIE is licensed under ``AGPL-3.0-or-later`` (Python and R). The AGPL
is a strong copyleft license: a modified MORIE that is distributed, or
offered to users over a network, must publish its source. The optional
Linux-kernel adjuncts (``kernel-module/morie.c`` and
``daemon/morie_lsm.py``) remain GPL-2.0-only because the kernel ABI
requires it; they are NOT part of the R / Python distribution. See the
``LICENSE`` file for the full text and ``LICENSING.md`` for the
per-component breakdown.

----

Documentation index
-------------------

If you prefer a single linear walkthrough rather than the sidebar
navigation, every page on this site is listed below — top to bottom:

- :doc:`learn/index` — From-zero tutorial track. Start here if you
  have never opened a Python or R console before.
- :doc:`install` — Installation instructions for Python, R, macOS,
  Linux, Windows, plus LLM provider setup.
- :doc:`cli` — Reference for every ``morie …`` subcommand.
- :doc:`hosted` — The hosted LLM tier, a last resort: requesting a key,
  storing it, the signed services document, limits, what is and is not
  stored.
- :doc:`methods/index` — Statistical-methods reference. Estimands,
  causal estimators, survey statistics, spatial methods, Hawkes
  processes, statistical physics of crime, OTIS / TPS / SIU
  pipelines, the MRM framework, key empirical findings.
- :doc:`api/index` — Python and R API reference
  (function signatures and docstrings).
- :doc:`contributing` — Development setup, test conventions,
  module-addition guide.

.. toctree::
   :maxdepth: 1
   :caption: Navigation
   :hidden:

   learn/index
   install
   cli
   hosted

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   architecture
   methods/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
   acknowledgments
