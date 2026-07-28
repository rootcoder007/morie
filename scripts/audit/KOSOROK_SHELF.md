# Kosorok shelf (batch 5) — 39 placeholders

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

**Remaining (28)**: ksr022 ksr025 ksr029 ksr031–ksr045 ksr050–ksr056
ksr059 ksr061 ksr062 ksr065 — entropy/bracketing, weak-convergence
characterisations, the functional delta method and its bootstrap,
Kaplan-Meier Hadamard derivatives, M-estimator expansions, LAN.

## Defects found

- `hadamard_derivative` first used a one-sided difference whose O(t)
  truncation error (9e-4 at t = 1e-4) swamped the convergence
  tolerance, so a clean quadratic reported non-convergence. Replaced
  with Richardson extrapolation, exact for quadratics. A central
  difference was rejected deliberately: the derivative is
  *directional*, and central differencing reports 0 for |·| at 0
  where the one-sided derivative is 1 — the test now pins both
  directions.
