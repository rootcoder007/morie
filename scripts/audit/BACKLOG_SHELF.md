# Non-DL hand-named backlog (batch 7 candidate) — 4,223 modules

Census 2026-07-29 (post-DL-shelf): every remaining hand-named module
whose body is the wrapped template `result = float(np.mean(...))`.
List: regenerate with the wrapped-template grep filtered to basenames
<= 13 chars. ZERO overlap with the DL shelf (verified) and with the
11,121 auto-extracted stubs (separate population, separate decision).

## Clusters by cited source

| Modules | Source |
|---|---|
| 381 | Advanced Statistics in Criminology (handbook) |
| 329 | MVSML (multivariate statistical machine learning text) |
| 208 | Rangayyan (beyond batch 1's 21 — biomedical signal analysis) |
| 179 | Use R! series |
| 73  | Wilcox (robust statistics) |
| 63  | Wasserman (All of Statistics / Nonparametric) |
| 32  | Ghosal & van der Vaart (Bayesian nonparametrics) |
| 29  | no reference line (triage individually) |
| 25 each | Silver et al (RL papers), Coles (extreme values) |
| ~23-11 each | Aitchison (compositional), D'Orsogna & Perc, Jumper
|     | (AlphaFold), Schabenberger Ch 4-6, Cochran (sampling), Hastie
|     | ESL, Boyd CVX, Anselin (spatial), Fauzi, Montesinos Lopez,
|     | Manski (partial identification), + long tail (~25 sources) |

## Proposed pipeline (NOT LAUNCHED — costed for approval)

Same as the DL shelf, which delivered 652 agent-drafted modules with
zero faked implementations surviving lead verification:
- Slice by book into ~75-module tranches -> ~56 drafting agent-runs.
- Measured DL-shelf agent cost: 200-450k tokens per ~70-module run.
- ESTIMATE: ~15-22M tokens drafting + ~4-6M for R parity + lead
  verification overhead. This is 3-4x the whole DL shelf.
- Book-as-spec applies: several sources are already in the library
  (Rangayyan, Schabenberger, Wasserman, Wilcox, Coles, Hastie, Boyd);
  the criminology handbook and MVSML need locating in books.csv or
  from Vee before their 710 modules can be PDF-verified.
- Do NOT launch without explicit approval and a per-phase budget from
  Vee ("I am not rich" is standing policy). Suggested phasing if
  approved: Phase 1 = sources already in the library with the
  strongest overlap to existing verified cores (Rangayyan 208,
  Wasserman 63, Wilcox 73, Schabenberger ~38, Hastie 13 -> ~400
  modules); Phase 2 = criminology handbook + MVSML after PDFs are
  confirmed; Phase 3 = long tail.

## The other population: 11,121 auto-extracted stubs

Separate decision, NOT implementation-by-default: the triage
(AUTOEXTRACTED_TRIAGE.md) marked most as formula-only with unclear
call contracts. Options: (a) delete them from the package as noise,
(b) keep as documented formula references without executable claims,
(c) implement the minority with clear contracts. Needs Vee's call —
they inflate the module count without carrying tested semantics.
