# Gibbons shelf (batch 4) — 68 nonparametric placeholders

Spec: Chakraborti, S. & Gibbons, J. D., *Nonparametric Statistical
Inference*, 5th ed. (CRC/Taylor & Francis). PDF in the library;
**printed page = PDF page − 21** (verified: printed 54 = PDF 75).
Module names encode chapter.section: gb1121 = Ch 11.2.1 (Kendall tau),
gb433 = Theorem 4.3.3 (Kolmogorov limit), gb2111 = Theorem 2.11.1
(tolerance intervals).

PDF-verified so far (transcribed, not recalled):
- Theorem 3.2.1 joint runs distribution, printed p. 77 (PDF 98)
- Theorem 4.3.3 K–S limit L(d) = 1 − 2 Σ (−1)^{i−1} e^{−2i²d²},
  printed p. 108–115 (PDF 129–136)
- Theorem 2.11.1 tolerance: U(s) − U(r) ~ Beta(s−r, n−s+r+1),
  printed p. 60–61 (PDF 81–82)
- Eq. 12.4.4 concordance W = 12S/(k²n(n²−1))

Oracles: scipy (kendalltau, spearmanr, mannwhitneyu, kruskal,
ks_1samp, kstwobign, chi2_contingency) wherever a scipy counterpart
exists — implement from the Gibbons formula, then cross-check.

## Clusters

| # | Cluster | Modules |
|---|---------|---------|
| A | Rank correlation + concordance (13) | gb1121 gb1122t gb1131n gb1131t gb1141 gb1241 gb1241t gb_kt2 gb_ktv gb_sp2 gb_spv gb_wcin gb_blt |
| B | Runs tests (11) | gb321 gb321c gb321l gb322 gb32l2 gb32l3 gb32lu gb32mn gb32vr gb331 gb332 gb34mn |
| C | Order statistics / EDF / coverages (16) | gb2111 gb2111c gb2112 gb221 gb2311 gb2313 gb232 gb233 gb2431 gb251 gb_eqf gb_lsm gb_med gb_pit2 gb_rng gb_rnk |
| D | K–S / goodness of fit (6) | gb433 gb434bt gb435 gb4351 gb_pp gb_qq |
| E | Two-sample ties + linear rank (10) | gb1041t gb661t gb661v gb821t gb_binmw gb_mw2 gb734 gb735 gb736 gb7381 |
| F | ARE + fundamentals + association (12) | gb1321 gb1323 gb_are1 gb_are2 gb_are3 gb_are4 gb_are5 gb_ar6 gb_ar7 gb_psi gb_c1 gb_c2 gb_cc gb_clt gb1421c gb1421t gb_cq |

## Status

- [x] A rank correlation (13) -- cluster test 9/9
- [x] B runs (12) -- exact vs enumeration, 4/4 files
- [x] C order statistics (16) -- 9/9 incl. Monte Carlo anchors
- [x] D K-S (6) -- vs scipy kstwobign/ksone
- [x] E two-sample/linear rank (10) -- ties vs scipy asymptotic
- [x] F ARE/fundamentals (17) -- Table 13.3.1 re-derived from densities
- [x] Legacy tests re-fixtured
- [x] Full run green: 179 tests, 4 cluster files + 74 legacy, 3.4s on L14
- [x] R collision scan + parity: R/gibbons_native.R, 69 tests green, mirrored both trees

## Findings during implementation

- **gb736 placeholder implied folded scores are always symmetric.**
  The book's Theorem 7.3.6 proof builds the conjugate by swapping the
  two halves (Z'_i = Z_{i+N/2}, PDF p. 282), which requires EVEN N.
  Enumeration confirms: N = 6, 8 give skewness 0; N = 7 gives 0.089.
  The module now refuses odd N with the measured counterexample in
  the error message.
- **gb_are5 placeholder claimed ARE(Mood, F | normal) = 3/pi.** The
  book's own derivation (Sec. 13.3.3, the e(M_N) calculation ending
  in "ARE(M_N, T_mn) = 15/(2 pi^2)", PDF-verified) gives 15/(2 pi^2)
  = 0.760. Fabricated constant replaced; the test asserts the true
  value AND asserts it differs from the fabricated one.
- Table 13.3.1 values are both hard-coded (as the book states them)
  and RE-DERIVED from the densities via the efficacy integrals in
  `_gb_are.efficacy_are`; the cluster test requires the two routes to
  agree to 1e-6 for all four distributions.
- Runs cluster is tested against brute-force enumeration of every
  arrangement (all C(n1+n2, n1) sequences), cell by cell -- Theorem
  3.2.1/3.2.2, the marginal, both run-length theorems, and the
  moments all match exactly.

## Why the combined run appeared to hang

`tests/fn` holds 36,310 files in one directory. pytest re-globs that
directory **once per file argument**, so passing 78 `tests/fn/*.py`
paths cost roughly 13 s each -- over 16 minutes with no output, which
read as a hang. Verified: one file 13.9 s, eight files 102 s (8x), and
the same 78 files copied into a scratch directory run in **3.4 s**.

`scripts/audit/run_fn_subset.sh` does the scratch-directory dance;
use it for any multi-file `tests/fn` run.

Running that way immediately surfaced one real failure the per-cluster
runs had not: `test_gb251` asserted a single seed's K-S p-value above
0.01, but under the true generator that p is Uniform(0,1) and so dips
below 0.01 about 1% of the time (seed 1 gives 0.0066). Rewritten as a
rate over 20 draws.
