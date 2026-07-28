# Kosorok shelf (batch 5) — 39 placeholders — **COMPLETE**

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
