# Armstrong / spatial-voting shelf (batch 2 of the book-as-spec series)

Triaged 2026-07-27. 73 modules cite Armstrong et al., *Analyzing
Spatial Models of Choice and Judgment*. The book has SIX chapters
(1 Introduction p.1, 2 Issue Scales p.13, 3 Similarities p.67,
4 Unfolding Rating Scale p.107, 5 Unfolding Binary Choice p.129,
6 Bayesian Scaling p.181 -- registry row verified 2026-07-26).
Citations to Ch 7-10 are fabricated.

## Census (L14, placeholder = mean/se template present)

- 30 placeholders needing real implementations
- 43 real implementations needing citation verification; 13 cite
  fabricated chapters: agrmt(7) alcov(9) algnm(8) bayid(9) btlmd(8)
  clfrt(10) gmpre(10) nmnlt(8) ocslt(7) prech(10) rollc(7) sptag(8)
  vtpwr(10)

## Acquired replacement primaries (books.csv, all verified 2026-07-26)

- Shapley-Shubik 1954 APSR 48(3) 787-792 -> vtpwr, agset
- Banzhaf 1965 Rutgers L Rev 19(2) 317-343 -> vtpwr
- Train 2009 Discrete Choice Methods 2e -> mnpbt
- Schonemann 1970 Psychometrika 35(3) 349-366 -> unfld (metric unfolding)
- Poole 2000 Political Analysis 8(3) 211-237 -> optcl, csphr
- Kaiser 1960 EPM 20(1) 141-151 + Cattell 1966 MBR 1(2) 245-276 -> dimrd
- Bailey 2007 AJPS 51(3) 433-448 -> brdgr
- Armstrong Ch 1-6 itself (txt + pdf in library) for the real chapters:
  sec 3.1 CMDS p.68, sec 3.2.2 Agreement Scores p.88, sec 5.4 OC p.156

## Cluster plan

1. **Fabricated-citation corrections** (13 real modules): verify the
   method against the correct acquired primary, replace the citation,
   fix any math the wrong spec hid, re-fixture tests.
2. **Voting rules** (Ch 1 cluster: borda, irv, appvl, stquo, mdnvt,
   qvote, rndut, eudst, ...): hand-checkable, placeholders first.
3. **MDS family** (Ch 3: mmds, smacf, krust, isotr, procs, mmdsf,
   nmdsf, shrpd, agrsc): Armstrong sec 3.1 + Kruskal/Shepard primaries
   as needed.
4. **Unfolding** (Ch 4: unfld, foldp, bymds + Schonemann 1970).
5. **Optimal classification + NOMINATE** (Ch 5 + Poole 2000: optcl,
   csphr, oclin, ocslt, wnoma, wnoml, wnomp, cutto, wghtm, bysid,
   pscrc, ricei, apre, agpar, brdgo, oclin).
6. **Bayesian/IRT ideal points** (Ch 6: bayam, bymds, bmdul, emtxt,
   hsirt, irtdq, irtid, mcmpp, pscli, dwnmn, brdgr).
7. **Remainder** (Ch 2 issue scales: ambtc, chopit, plpol, mdvtr,
   rcall; misc: dimrd, mnpbt, polrz, tmort, agset, amfit, mdspl,
   citym, sptag...).

Rules carried over from the causal batch: PDF is truth, txt is index;
verify every citation before use; rates over seeds; assertions a stub
cannot pass; three-way parity (Python + morie/R + rmorie); push only
feat/native-specializations; collision-scan R filenames AND function
names before mirroring.

## Status: COMPLETE (2026-07-27)

Full tests/fn sweep after the batch: 0 FAILED, EXIT=0 on L14
(~/fnrun/armstrong_full.log). All 32 placeholders real, all 13
fabricated citations corrected, R parity closed (33 of 35 mirrors
pre-existed; 2 gap-fillers added to both trees).

## Done

- **R parity (batch 2): complete via discovery, not duplication.** The
  filename+function collision scan found R/spatial_voting.R already
  carrying 35 functions -- MDS/SMACOF/nonmetric/Procrustes, the whole
  NOMINATE family incl. loglik and bootstrap, Bayesian AM/MDS/
  unfolding, CJR/EM/dynamic IRT, wordfish, anchoring vignettes, OC --
  plus Rice cohesion in R/algnm.R and APRE already present. Only two
  mirrors were genuinely missing and were added as
  R/spatial_voting_native2.R in both trees: morie_party_unity (CQ
  variant included) and morie_heteroskedastic_scales (Lauderdale,
  Political Analysis 18(2) 151-171). 10/10 testthat on L14.

- **Clusters 2/4/6/7, IRT-Bayesian + survey group (13/13 -- closes all
  32 placeholders).** ambtc (A-M bootstrap SEs, replicates sign-aligned
  before the sd), bayam (Gibbs BAM sampler, Hare et al. 2015 AJPS
  59(3) 759-774; stimulus recovery r > 0.99), irtdq (quadratic-utility
  probit -- the test caught a SIGN ERROR in my first derivation:
  U_yea - U_nay = 2(z_y - z_n)(x - mid), not (z_n - z_y)), irtid
  (identification normalisation with polarity/pivot feasibility
  checks), foldp (single-peakedness diagnostic, Armstrong Ch 4),
  plpol (plot data layer, Sec 5.3.5.1), mcmpp (Albert 1992
  data-augmentation Gibbs for probit IRT, JEBS 17(3) 251-269; order
  recovery r > 0.9), pscli (rollcall screen -> Gibbs, ideal()-style),
  hsirt (Lauderdale heteroskedastic scales -- placeholder cited AJPS,
  the paper is Political Analysis 18(2) 151-171: another wrong venue
  caught), emtxt (wordfish, Slapin-Proksch 2008 AJPS 52(3) 705-722;
  theta recovery r > 0.95), bymds (Bakker-Poole 2013 PA 21(1) 125-140
  RW-Metropolis, draws Procrustes-aligned), bmdul (Bayesian unfolding
  over X and Y jointly), chopit (King et al. 2004 APSR 98(1) 191-207;
  DIF shift 0.8 recovered; location anchored at mean vignette level
  = 0 after the sampler exposed the tau-mu location
  non-identification by drifting both to -62). All primaries verified
  before use; legacy tests re-fixtured; 39/39 green on L14.

- **Cluster 5 core, NOMINATE/OC group (8 placeholders).** wnomp
  (Gaussian-utility choice probability -- the NOMINATE signature vs
  quadratic IRT; equidistant = 0.5 asserted), wnoml (Bernoulli
  log-likelihood + the Sec 5.3.5 fit statistics from the same
  probabilities), ricei (Rice 1925 PSQ 40(1) 60-72, verified),
  apre (the p.143 APRE formula, minority-vote fixture), agpar (party
  unity with the CQ unity-votes-only variant), oclin (Poole 2000
  exhaustive cutpoint search along the discriminant; zero errors on
  separable fixtures in 1-D and 2-D), pscrc (recode + lopsidedness
  screen, wnominate's lop = 0.025 default), brdgo (bridge alignment =
  centre + scale + Schoenemann rotation on shared actors; exact
  recovery of a planted similarity transform; Bailey 2007 as the
  bridging primary). Legacy tests re-fixtured for all 8; 24/24 green
  on L14.

- **Cluster 3, MDS family + spatial utilities (11/11 placeholders).**
  mmdsf (Torgerson 1952 double centering, exact recovery of planted
  Euclidean configurations asserted), krust (Kruskal 1964 stress-1 +
  the verbal scale), isotr (PAV monotone regression, hand case
  [1,3,2,4] -> [1,2.5,2.5,4]), smacf (de Leeuw 1977 Guttman-transform
  majorization; the never-increasing stress path is asserted, weighted
  variant included), nmdsf (Kruskal's alternation; stress < 0.05 on
  rank-preserving cubic distortion of planted distances), shrpd
  (Shepard 1962 diagram data + PAV trend + Spearman rho), procs
  (Schoenemann 1966 closed form; undoes a planted rotation+reflection
  to 1e-10), agrsc (front-end to agrmt, Armstrong Sec 3.2.2), eudst
  (quadratic utility, scalar and matrix forms), rndut (McFadden 1974
  closed-form logit == softmax asserted; simulated probit for normal
  errors, Train 2009 Ch 3/5), stquo (proposal-vs-status-quo majority,
  median-voter check). Citations verified by web search 2026-07-27
  (Kruskal 29(1) 1-27 + 29(2) 115-129, Torgerson 17(4) 401-419,
  Schoenemann 31(1) 1-10, McFadden pp. 105-142). Legacy tests
  re-fixtured for all 11. Verified locally (Mac, pure numpy/scipy);
  L14 re-verification pending a Tailscale re-auth.

- **Cluster 1, fabricated citations (13/13).** vtpwr, algnm, sptag were
  already corrected in the 07-26 session. This session fixed the other
  ten: agrmt -> Sec 3.2.2 p.88; rollc -> Ch 5 + Poole-Rosenthal 1997;
  clfrt/prech/gmpre -> the Sec 5.3.5 footnote p.143 (PDF page verified;
  prech and gmpre match the printed APRE and GMP formulas exactly, so
  citation-only fixes); nmnlt -> Poole-Rosenthal 1997 + Sec 5.3;
  ocslt -> Poole 2000 (library PDF) + Sec 5.4; bayid ->
  Clinton-Jackman-Rivers 2004 APSR 98(2) 355-370; btlmd ->
  Bradley-Terry 1952 Biometrika 39 324-345 + Hunter 2004 Ann Stat
  32(1) 384-406; alcov -> Kruschke 1992 Psych Rev 99(1) 22-44 (all
  external citations verified by web search 2026-07-27).
  New tests/fn/test_armstrong_fitstats.py pins the footnote formulas
  with hand fixtures (APRE 0.75 case, GMP exp-mean-loglik case,
  Sec 3.2.2 agreement matrix).
