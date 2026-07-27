# Econometrics / volatility shelf (batch 3 of the book-as-spec series)

Triaged 2026-07-27. 66 placeholder modules across four families. Books
acquired 2026-07-26 (registry rows): Tsay 2010 3e, Hamilton 1994/2020,
Brooks 3e/4e Python guide, Czado 2019 (13 page-range volumes),
RiskMetrics 1996 (lambda = 0.94 verified at PDF p.51).

## Families

1. **Realized/range volatility (20)**: volrm volsd volrv volbpv volrk
   volrs volyz volhar volhar1 volharj volraq voldoc voltsr voljr
   volpow volmuk volopn volsk volrls volrlmt
2. **GARCH extensions (~20)**: egarch egarcm garchm igarcm tgarcm
   mgrch volbekk volcgar volegar volfig volgar volgargd volgargt
   volges volgjr volgo volgvi volign volmsg volnsig voltgr volaprch
   (Tsay Ch 3 as spec; egrch/tgrch/archm/dccmd already real from the
   reds work)
3. **Cointegration (~8)**: egcoin engrgr johbu joholt johw vecmod
   mstrn (Hamilton/Tsay Ch 8; johsn/vecmf/coint already real)
4. **Copulas (~18)**: copgau copt copcla copfr copfra copgmb copjoe
   copod copExt copynm blncop clyfr taukcp spcoef plkt ginicop zxcp*
   (Czado 2019 as spec)

## Verified primaries (web, 2026-07-27)

- Corsi 2009 J Fin Econometrics 7(2) 174-196 (HAR-RV)
- Rogers-Satchell 1991 Ann Appl Prob 1(4) 504-512
- Yang-Zhang 2000 J Business 73(3) 477-492
- Barndorff-Nielsen & Shephard 2004 J Fin Econometrics 2(1) 1-48
  (power/bipower variation)
- Zhang-Mykland-Ait-Sahalia 2005 JASA 100(472) 1394-1411 (TSRV)
- Parkinson 1980 J Business 53(1) 61-65
- Garman-Klass 1980 J Business 53(1) 67-78
- RiskMetrics 1996 p.51 (registry, PDF-verified)

Rules as before: PDF is truth; verify before cite; rates over seeds;
assertions a stub cannot pass; three-way parity with R
collision-scanned by filename AND function name; push
feat/native-specializations only.

## Done

- **Family 1, realized/range volatility (20/20).** volrm (RiskMetrics
  EWMA, lambda = 0.94 from the PDF-verified p.51), volsd, volrv,
  volbpv (BNS 2004; jump-robustness asserted: a 0.2 jump lifts RV by
  0.04 and BPV by < 0.01), volrk (Bartlett realised kernel; default
  bandwidth switched to ceil(m^(1/3)) after measurement -- sqrt(m)
  over-corrected on 2/10 seeds, cube root wins 10/10), volrs
  (Rogers-Satchell 1991, flat-bar zero + hand value), volyz
  (Yang-Zhang 2000, k formula pinned + sigma recovery on simulated
  OHLC), volhar/volhar1/volharj (Corsi 2009, BPQ 2016 HAR-Q, ABD 2007
  HAR-RV-J -- nesting R2 inequality asserted), volraq (QV +
  quarticity), voldoc (BNS decomposition with zero-truncation),
  voltsr (ZMA 2005; noise-correction as a rate over seeds), voljr
  (Mancini threshold), volpow (BNS power variation; mu_2 = 1 identity
  pins p = 2 == RV), volmuk (subsample-averaged kernel), volopn
  (Black-Scholes implied vol round-trip to 1e-8 + no-arbitrage bound
  errors), volsk (Harvey-Ruiz-Shephard QML filter; regime shift
  tracked), volrls (nowcast-vs-forecast convention documented against
  volrm), volrlmt (ABDL 2003 log-RV AR(1) with lognormal
  back-transform). All primaries verified; legacy tests re-fixtured;
  58/58 green on L14.

### Family 2 -- copulas (19 modules) -- DONE 2026-07-27

Spec: Czado (2019), *Analyzing Dependent Data with Vine Copulas*,
Ch. 2-3. **Table 3.2, p. 54 read in the PDF** for every
parameter/Kendall's-tau relation used (Gaussian/t `(2/pi) arcsin rho`;
Gumbel `1 - 1/delta`; Clayton `delta/(delta+2)`; Frank's Debye-function
form; Joe's digamma form). Theorem 3.9 eq. (3.17)-(3.18) for the
general Archimedean and extreme-value integrals; Table 3.1 p. 52 for
the Pickands functions.

New shared core `src/morie/fn/_copula.py`: `copula_cdf`, `copula_tau`,
`tau_to_theta`, `FAMILIES`.

Modules: copgau copt copcla copgmb copfra copjoe plkt (family CDFs);
taukcp spcoef blncop ginicop (dependence measures); copExt (Pickands /
extreme-value); clyfr copfr (survival copulas on Kaplan-Meier
margins); copod (Li et al. 2020 outlier detection); zxcpc zxcpg zxcpv
(pairwise-tau fits to multivariate data).

Tests: `tests/fn/test_copula_cluster.py` (16) + 18 rewritten legacy
files = 52 green on L14. The cluster asserts the copula *axioms* on
every family (grounded margins, uniform margins, Frechet-Hoeffding
bounds, 2-increasing rectangle mass), max-stability for the
extreme-value branch, and closed-form vs numeric-double-integral
agreement for tau -- assertions no placeholder can pass.

Design notes worth keeping:
- `zxcpc` returns NaN for pairs whose sample tau is negative rather
  than clamping: Clayton cannot represent negative dependence and
  hiding that would be a silent wrong answer.
- `spcoef` uses the exact elliptical `(6/pi) arcsin(rho/2)` for the
  Gaussian and quadrature elsewhere, reporting which route it took in
  an `exact` flag.
- `copynm` is NOT a copula (copy-number-variant detection, genomics);
  it was swept in by the name match and is out of scope for this
  shelf.
