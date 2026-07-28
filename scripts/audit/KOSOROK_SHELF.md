# Kosorok shelf (batch 5) — 39 placeholders — **COMPLETE (Python + R)**

Spec: Kosorok, M. R. (2008), *Introduction to Empirical Processes and
Semiparametric Inference*, Springer. **Filed in the library under its
ISBN** (`pdf/978-0-387-74978-5.pdf`), not its title.

PDF-verified anchors:
- Brownian bridge covariance `F(s ∧ t) − F(s)F(t)` (Ch. 2)
- LIL eq. (2.21): `limsup ‖G_n‖_∞ / √(2 log log n) ≤ 1/2` a.s.
- Chung liminf companion: `liminf √(2 log log n)·‖G_n‖_∞ = π/2` a.s.

## Approach

Most of this book states theorems, not procedures. Each module
therefore returns the finite-sample **witness** of its theorem — the
quantity the theorem bounds, computed on real data — rather than a
bare `True`. A "theorem module" that no input can falsify is exactly
as untestable as the placeholder it replaces.

Shared core: `src/morie/fn/_kosorok.py` — `empirical_df`,
`empirical_process`, `bridge_cov`, `sup_norm`,
`bootstrap_multiplier_process`, `bracketing_number_monotone`,
`covering_number_grid`, `hadamard_derivative`, `cox_score`.

## Progress

**Tranche 1 (11 modules) — DONE, 13/13 tests green**

ksr020 ksr024 (Ch. 1 models), ksr026 ksr027 ksr028 ksr030 ksr058
ksr060 (Ch. 2 empirical process), ksr063 ksr068 ksr071 (Ch. 3
semiparametric efficiency).

Verified against simulation, not just self-consistency: the bridge
covariance matches the Monte Carlo covariance of `G_n` to ±0.05; the
Cox efficient information matches the sampling standard deviation of
β̂ across 25 replications; the multiplier bootstrap reproduces the
bridge covariance structure; the U-statistic matches brute-force
enumeration.

**Tranche 2 (10 modules) — DONE, 21/21 tests green**

ksr029 ksr031 ksr032 ksr033 ksr034 ksr035 ksr036 ksr037 ksr038 ksr039
— class-indexed GC, tightness/equicontinuity, the weak-convergence
characterisation, uniform and bracketing entropy, both Donsker
theorems, the bounded-Lipschitz metric.

Tested by the SEPARATION each theorem draws, not just its easy side:
polynomial vs exponential bracketing growth (finite vs diverging
entropy integral), finite entropy with a non-integrable envelope
(fails GC), a smooth process vs a shrinking spike (tight vs not),
matched marginals vs a rescaled law (fidi passes vs fails).

**Tranche 3 (10 modules) — DONE, 30/30 tests green**

ksr022 ksr025 (Ch. 1), ksr042 ksr050 ksr051 ksr052 ksr053 ksr055
ksr056 ksr059 — the functional delta method, Frechet differentiability,
continuous invertibility, both Kaplan-Meier Hadamard derivatives,
M-estimator expansions, the LAD Lipschitz bound and the KMT bound.

**Tranche 4 (8 modules) — DONE**

ksr040 ksr041 ksr043 ksr044 ksr045 ksr061 ksr062 ksr065 — the two
bootstrap Donsker characterisations, the quantile Hadamard sandwich
and its first-order collapse, the bootstrap delta method,
differentiability in quadratic mean, pathwise differentiability and
the efficient influence function via the information operator.

## COMPLETE — 39/39 modules, 192 tests green

All four cluster files plus 39 re-fixtured legacy tests run in 3.3 s
via `scripts/audit/run_fn_subset.sh`.

## Defects found

- `hadamard_derivative` first used a one-sided difference whose O(t)
  truncation error (9e-4 at t = 1e-4) swamped the convergence
  tolerance, so a clean quadratic reported non-convergence. Replaced
  with Richardson extrapolation, exact for quadratics. A central
  difference was rejected deliberately: the derivative is
  *directional*, and central differencing reports 0 for |·| at 0
  where the one-sided derivative is 1 — the test now pins both
  directions.
- `ksr032` first judged finite-dimensional convergence against a fixed
  0.15 threshold. At 400 replications over 40 grid points two samples
  from the **same** law already differ by 0.21 in mean and 0.25 in
  variance, so the constant rejected identical distributions. The
  tolerance is now derived from the replication count and grid size
  (`sigma sqrt(2/n_rep) sqrt(log k)`), and both the gap and the
  tolerance are returned so the comparison is auditable.
- `ksr050` (Frechet) first used a numerical derivative recomputed in
  each perturbation direction. That is the *Hadamard* derivative, so
  the check was vacuous — it reported `|·|` at 0 as Frechet
  differentiable with a residual ratio of exactly 0. Frechet requires
  a single linear map valid in every direction; the fallback now
  builds the Jacobian once and applies it linearly, and the kinked map
  correctly shows a ratio pinned at 1.
- `ksr053` refuses to integrate when `L(u⁻)S₀(u⁻)` vanishes. That
  surfaced a modelling error in my own test, not a code bug: I passed a
  hazard-like `L(u) = 0.5u`, which is 0 at the origin, where the
  Kaplan-Meier `L` is an at-risk probability and positive there. The
  guard stays; the test now uses a proper at-risk `L` and separately
  asserts the refusal.

## A SECOND placeholder template, found via the warnings

Chasing the 9 pytest warnings turned up `ksr064` and `ksr069`, which
compute

```python
estimate = np.median(beta)
se = 1.2533 * np.std(beta, ddof=1) / np.sqrt(n)
```

`beta` is a scalar, so `np.std(..., ddof=1)` divides by zero degrees
of freedom and **`se` is silently NaN**. Worse, both functions ignore
`Z`, `V` and `d` entirely while their docstrings state the Cox partial
likelihood.

This is a **different placeholder template** from the
`result = np.mean / se = np.std` one the earlier census counted.
Repo-wide: **333 modules** carry it, with **zero overlap** with the
16,182 template-A placeholders. The true hand-named placeholder count
is therefore higher than previously reported by 333.

`ksr064` and `ksr069` are now real: the Cox partial likelihood of
eq. (3.4) and the Breslow baseline, with standard errors from the
observed information. The remaining 331 template-B modules are
unaudited and should be swept like template A.

## R parity — added after the fact

The shelf was marked COMPLETE when only the Python side existed. That
was wrong; the standing rule is that nothing stays Python-only. Fixed
now.

Collision scan first, and it mattered: R already carried **twenty**
`morie_ksrNN_*` functions (empirical process, Donsker class,
Glivenko-Cantelli, VC dimension, bracketing number, maximal
inequality, both bootstraps, Z- and M-estimators, efficient score,
information bound, tangent space, profile likelihood, one-step
estimator, influence function, counting process, Nelson-Aalen, Cox
partial likelihood, censoring survival), plus eighteen `morie_dsp_*`
signal-processing functions.

`R/kosorok_native2.R` therefore adds only the genuinely absent
surface: bridge covariance, the exact sup norm, LIL and Chung
constants, the KMT bound, U-processes, the entropy integral with both
Donsker/GC envelope conditions, the functional delta method, the
Frechet check, the Kaplan-Meier Hadamard derivative, DQM, the quantile
sandwich, the BL metric and the tightness check. 42 tests green.

**One R-only bug the cross-language anchor caught:**
`morie_functional_delta` used the raw Jacobian where the delta method
needs it APPLIED to the observed deviation, matching
`morie.fn.ksr042`'s directional derivative. The remainder came out as
−395.99 instead of 0.01 — a factor of 1/deviation. Both the
directional derivative and the Jacobian are now returned separately.
