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

(nothing yet)
