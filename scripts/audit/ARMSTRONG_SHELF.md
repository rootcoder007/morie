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

## Done

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
