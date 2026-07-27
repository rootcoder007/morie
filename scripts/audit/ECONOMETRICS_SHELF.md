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

### Family 3 -- GARCH extensions (22 modules) -- DONE 2026-07-27

Spec: Tsay (2010), *Analysis of Financial Time Series*, 3rd ed.,
Ch. 3, read in the library PDF. **Printed page = PDF page - 27.**
Chapter index (was marked TO BE POPULATED in books.csv):

| Section | Topic | Printed p. | PDF p. |
|---|---|---|---|
| 3.4 | ARCH model | 115 | 142 |
| 3.5 | GARCH model | 131 | 158 |
| 3.6 | Integrated GARCH | 140 | 167 |
| 3.7 | GARCH-M (eq. 3.23, Table 3.2) | 142 | 169 |
| 3.8 | Exponential GARCH (eq. 3.24-3.25) | 143 | 170 |
| 3.9 | Threshold GARCH (eq. 3.34) | 149 | 176 |
| 3.10 | CHARMA (eq. 3.36) | 150 | 177 |
| 3.12 | Stochastic volatility | 153 | 180 |
| 3.13 | Long-memory SV | 154 | 181 |

Equations transcribed from the PDF, not recalled: the IGARCH weight
constraint alpha = 1 - beta (p. 141); the EGARCH weighted innovation
g(e) = theta e + gamma(|e| - E|e|) with E|e| = sqrt(2/pi) for a
Gaussian (the Remark on p. 143); the TGARCH indicator form of
eq. (3.34).

New shared core `src/morie/fn/_garch.py`: `garch_recursion`,
`garch_fit` (Gaussian/t/GED QML on an unconstrained
reparameterisation), `garch_forecast`, `bekk_fit`, `ms_garch_fit`,
`var_es`; specs garch igarch egarch gjr tgarch aparch cgarch figarch.

Modules: garchm volgar igarcm volign egarch egarcm volegar tgarcm
volgjr voltgr volaprch volcgar volfig volgargt volgargd (univariate
fits); mgrch volbekk (BEKK); volgo (orthogonal); volmsg
(Markov-switching); volgvi volges (VaR / expected shortfall);
volnsig (EGARCH + skew-GED).

Tests: `tests/fn/test_garch_cluster.py` (19) + 22 rewritten legacy
files. The cluster checks the recursions against the Tsay equations
computed by hand, parameter recovery as rates over seeds, that
standardised residuals actually remove the ARCH effect (LM statistic
falls) rather than merely being finite, BEKK positive-definiteness at
several t, and that the forecast converges to omega/(1 - a - b).

Two bugs my own tests caught:
- The persistence lookup was written as a dict literal, which
  evaluates every branch: a KeyError on 'gamma' fired for specs that
  have no gamma. Replaced with an if/elif chain.
- The Markov-switching transition matrix pinned column 0 rather than
  the diagonal for identification, so regime 0's self-transition
  collapsed to 3e-14 -- a "regime" that never persists. Fixed by
  pinning the diagonal logit and starting the off-diagonals negative.

One wrong constant I wrote from memory and then derived instead: the
normal 5% expected shortfall is 2.0627128075074253 (phi(z)/alpha), not
the 2.0627128425 I first typed.

### Family 4 -- cointegration + four misfiled modules (7) -- DONE 2026-07-27

Spec: Hamilton (1994) *Time Series Analysis* Ch. 19-20 and Tsay Ch. 8
for cointegration; Hyndman & Athanasopoulos *FPP* 3rd ed. for the
forecasting modules.

**Four of the seven were name-match false positives.** `johbu`,
`joholt` and `johw` are not Johansen anything -- they are "joseph"
hierarchical reconciliation and Holt / Holt-Winters smoothing; `mstrn`
is a multistate transition matrix (Aalen-Johansen), a survival method.
They were implemented against their real sources rather than forced
into the cointegration frame.

New core `src/morie/fn/_coint.py`: `adf_test`, `engle_granger`
(MacKinnon 2010 critical values, p-value *bands* rather than a
false-precision interpolation), `johansen` (reduced-rank trace test,
Osterwald-Lenum 1992 CVs), `vecm_fit`.

Modules: egcoin engrgr vecmod (cointegration); joholt johw (Holt and
Holt-Winters, with damped trend and a multiplicative form that refuses
non-positive data); johbu (bottom-up plus OLS/WLS optimal-combination
reconciliation, Wickramasuriya et al. 2019); mstrn (Aalen-Johansen
product integral).

Tests: `tests/fn/test_coint_cluster.py` (9) + 7 rewritten legacy files
= 23 green. Rates over seeds throughout: ADF rejects 8/8 stationary
series and 0/8 random walks; Engle-Granger finds 8/8 genuine
cointegrations and 0/8 spurious ones; Johansen recovers rank 1 on 6/6
cointegrated systems and rank 0 on 6/6 independent pairs.

One real bug my own test caught: Holt-Winters initialised its seasonal
indices from raw per-position period means, which absorb the
within-period trend and come out phase-shifted. Replaced with a
classical centred-moving-average detrend (Hyndman Sec. 3.4); the
seasonal-shape correlation went from 0.58 to 0.95. One test-side error
too: I compared the forecast to the seasonal pattern after subtracting
only its mean, leaving the linear trend in and capping the correlation
near 0.75 regardless of implementation -- the forecast has to be
detrended before the comparison means anything.

