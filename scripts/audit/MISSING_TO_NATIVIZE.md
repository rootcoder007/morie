
## Missing R twins (parity gap, NOT "Python-only OK")

Per Vee 2026-07-30: a Python fn module with no R counterpart is a
MISSING PORT, not an acceptable Python-only. Each needs an R version in
both trees for true three-way parity. Surfaced by the verification
sweep as "no R twin"; confirmed against parity_check.py (287 of 36,459
Python modules currently have BOTH R twins, so the vast majority are
un-ported -- this list is only the ones the sweep touched and fixed,
where correctness is now settled and the port can be written against a
known-good reference).

Confirmed-correct Python, R twin to be written:
- bshrk  (horseshoe Gibbs, Makalic & Schmidt 2016 -- now source-faithful)
- empby  (parametric EB, Morris 1983 -- now shrinks to estimated mean)
- eslsmt (Reinsch smoothing spline -- inverse now restored)
- dppca, vlfctn, otmapnk (fixed this sweep)
- and the broader set of ~36,000 fn modules with no R counterpart

This is a large standing backlog, not a same-day task. Recorded so it is
not mistaken for done.
