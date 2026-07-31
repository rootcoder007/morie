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


## Census correction 2026-07-29: 656 of the 4,223 are auto-extracted

The "hand-named backlog" slice was not clean. Splitting by function
name (`*_chapter_N_unnumbered_M`, `*_eq_N`) gives:

- **3,567 genuinely hand-named** modules -> `backlog_real.txt`
- **656 auto-extracted equation fragments** -> `backlog_autoextracted.txt`

The whole Wilcox "shelf" (73/73) is auto-extracted: names like
`wilcox_chapter_4_unnumbered_22`, bodies that are a single lifted
equation with no call contract. These belong to the 11,121
auto-extracted population, whose disposition (delete / keep as
documented formula references / implement the minority with clear
contracts) is Vee's decision and NOT implement-by-default. Phase 1's
real content is therefore Hastie 68 + Schabenberger 69 + Rangayyan 450
= 587, with Wasserman 66 already complete.

Practical rule for every future slice: filter the worklist through the
name test before costing or launching anything, or the estimate silently
includes fragments that should not be implemented at all.

## Census correction 2026-07-31: the name test must run on the FUNCTION name

The 2026-07-29 correction applied the auto-extracted name test to the
*filename*. Filenames are frequently short or hashed
(`msm111.py`, `..._wit2u10.py`) while the marker lives in the exported
function name. Re-running the test against `__all__` instead moves a
large slice from "real work" to "auto-extracted":

| Source | Phase-1 estimate | Actual hand-named |
|---|---|---|
| Rangayyan | 450 | **450** (confirmed) |
| Schabenberger | 69 | **69** (confirmed) |
| Hastie | 68 | **0** |
| Wilcox | 73 | 0 (already known) |
| Wasserman | 66 | 0 — complete |

All 213 Hastie-citing placeholders are auto-extracted. Most are not even
ESL: they cite James/Witten/Hastie/Tibshirani *ISLR*, a different book,
via `..._chapter_N_unnumbered_M`. Two independent tells confirm it
without opening the PDF — the docstring says "auto-extracted", and the
"Formula" field holds unparsed OCR
(`tstatistic for H0 : β=0t a k e st h ef o r m...`).

Whole-corpus recount, placeholder-bodied modules:

- **15,040** total
- **11,524** auto-extracted (NOT implement-by-default — Vee's call)
- **3,516** genuinely hand-named — the real backlog

Hand-named work not in the Phase-1 plan, by source: Morin *Probability*
174, criminology handbook 120, Brus *Spatial Sampling with R* 97,
*Analysis of Categorical Data with R* 69, Coles 29, Ghosal & van der
Vaart 25, no-reference 25, Aitchison 24. Full per-source worklist:
`scripts/audit/backlog_real_by_source.json` (1,863 sources).

So Phase 1 as scoped (Hastie 68 + Schabenberger 69 + Rangayyan 450 =
587) is really **519**, and the whole hand-named backlog is 3,516 rather
than the ~3,567 estimated — but distributed across far more books than
the plan assumed, most of which still need PDF confirmation before any
of their modules can be book-certified.

## Schabenberger shelf (69) — citation audit 2026-07-31

All **51** section-level citations verify against the actual PDF table of
contents of Schabenberger & Gotway (2005), *Statistical Methods for
Spatial Data Analysis*. Zero fabricated sections.

Of the 18 chapter-only citations, six named things the book does not
contain. Each was checked by reading the page, not by inference, and
replaced with a source verified against Crossref or against a printed
bibliography in the corpus:

| module | was | now |
|---|---|---|
| `spmidw` | "Schabenberger supplement" | Bivand, Pebesma & Gómez-Rubio (2013) §8.3.1, p. 215. In S&G "inverse distance" occurs only in the subject index. |
| `spsdm` | Schabenberger Ch 6 | Bivand et al. (2013) §9.4.2, pp. 307–311. In S&G "Durbin" occurs only in the reference list. |
| `spthom` | Schabenberger Ch 3 | Marjorie Thomas (1949) *Biometrika* 36(1–2):18–25, doi:10.1093/biomet/36.1-2.18. All four "Thomas" hits in S&G are the people Thomas Mueller / Thomas A. Louis. |
| `spmsim` | Schabenberger Ch 6 | Fotheringham, Yang & Kang (2017) *Annals AAG* 107(6):1247–1265, doi:10.1080/24694452.2017.1352480. MGWR postdates the 2005 book; "multiscale" has 0 hits. |
| `spgwrb` | Schabenberger Ch 6 | Bivand et al. (2013) §9.4.3, p. 318 (Gaussian kernel, leave-one-out CV). The **AIC selector claim was dropped** — no source here states it. |
| `spgwrk` | Schabenberger Ch 6 | Brunsdon, Fotheringham & Charlton (1996), doi:10.1111/j.1538-4632.1996.tb00936.x; Fotheringham et al. (2002). **bisquare / tricube / boxcar are attributed to the spgwr and GWmodel implementations, not to a book** — the only corpus hit for "bisquare" is Tukey's robust loss, and "tricube" has none. |

S&G *does* cover GWR, at pp. 316–317 — confirmed directly and
cross-confirmed by Bivand et al. (2013) p. 318 citing exactly that range.
So the GWR modules keep a legitimate secondary pointer to §6.1.3.

### Do not rename these six

The obvious cleanup — dropping the misleading `schabenberger_` prefix —
collides with existing exports: `spatial_durbin_model` (`sdurbm.py`),
`thomas_process` (`sgthm.py`, `ptthm.py`), `gwr_bandwidth` (`xrgwb.py`).

### Duplicate check before drafting

Several shelf entries duplicate modules that are **already implemented**:

- `spsdm` ↔ `sgdbn.py` (98 lines, implemented) and `sdurbm.py`
- `spthom` ↔ `sgthm.py` (70 lines, implemented)
- `spblkk` ↔ `spblk.py` `spatial_block_kriging` (127 lines)
- `spcokr` ↔ `cokrg.py` `cokriging` (110 lines)
- `spnst` ↔ `nstat.py` `nonstationary_covariance` (79 lines)
- `sptrs` ↔ `sptrn.py` `spatial_trend_surface` (83 lines)

(`splfun`→`boyd_dual_function` and `spblup`→`gblup_full` are false
matches from substring overlap; `spperiod`→`rangayyan_periodogram` needs
a judgement call — signal vs lattice periodogram.)

Note also that `sgdbn.py` and `sgthm.py` carry the **same false
Schabenberger attribution** in already-implemented code, so the
misattribution is not confined to this shelf. The wider `sg*` family
needs the same audit.

Implementing the duplicated entries would add redundant public API.
Decide dedup before drafting.

## Dedup + package-wide Schabenberger audit, 2026-07-31

### Package-wide citation audit result

Every Schabenberger citation in `src/morie/fn` was checked, not just the
69-module shelf. **2,195 modules cite the book.**

- **63 / 63 section-level citations verify** against the real TOC.
  (`sarla.py` and `sarre.py` cite §6.2.2.1, which a first pass flagged as
  missing — the section exists, "Simultaneous Autoregressive (SAR)
  Models"; the TOC regex only captured three levels. False positive from
  the audit, not a bad citation.)
- Chapter-only citations were screened by checking whether any
  distinctive title token appears in the book. **The screen must exclude
  the back matter** — the reference list starts at page index 465 — or
  bibliography and subject-index hits count as coverage and mask exactly
  the failures we were looking for. Against body text only, one
  module fell out: `sgdbn.py`, which is *implemented and shipped*.
  Fixed to LeSage & Pace (2009) doi:10.1201/9781420064254 + Bivand §9.4.2.

`spblk.py` also cited §5.2; block kriging has its own section, §5.7.1,
now cited as extending the §5.2 system.

### Only four of the suspected duplicates were real

Reading both sides changed the count from six to four:

- `spthom` is **not** a duplicate of `sgthm` — it computes the
  theoretical K-function, `K(r) = pi r^2 + mu[1-exp(-r^2/4 sigma^2)]/rho`;
  `sgthm` simulates the process. Complementary.
- `spnst` is **not** a duplicate of `nstat` — `spnst` is §8.2.1
  parametric models, `nstat` is the moving-window approach (§8.3.1, which
  is `spmwst`'s territory).

The four real ones now delegate instead of carrying a second
implementation: `spblkk`→`spblk`, `spcokr`→`cokrg`, `sptrs`→`sptrn`,
`spsdm`→`sgdbn`.

### Their tests were asserting the placeholder's fake contract

`test_sptrs.py` passed a 100-element random normal array as
`poly_degree`. All four pairs were like this, and all four now compare
against the implemented sibling.

**`str()` is not a usable comparison here.** `RichResult` is a dict
subclass whose `__str__` is empty unless built with a title, so
`str(got) == str(ref)` compares `'' == ''` and passes no matter what. The
tests use an array-aware deep compare on the payload, plus a
must-differ assertion so a degenerate comparator cannot pass.
