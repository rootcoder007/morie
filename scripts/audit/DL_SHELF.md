# DL/LLM shelf (batch 6) — 857 hand-named placeholders

Census 2026-07-28: modules citing the DL shelf books AND carrying the
wrapped template-A body `result = float(np.mean(...))` (the earlier
census signature `result = np.mean` missed the `float(...)` wrapping;
these are hand-named short modules, ZERO overlap with the 16k
auto-extracted stubs). Worklist: `find src/morie/fn -name '*.py' -exec
grep -l 'result = float(np.mean(' {} +` intersected with the book
citation grep — regenerate, never trust this file's counts after edits.

Books (all in the library; verify page maps before citing):
- `hm*` (329) + `gr*` (201): Géron, *Hands-On Machine Learning with
  Scikit-Learn and PyTorch* (2026 ed.)
- `km*` (243): Kamath, Keenan, Somers, *Large Language Models: A Deep
  Dive* (2024) — modules encode Ch/Eq/page (km001 = Ch 2 Eq 2.1 p.30)
- `al*` (53): Alammar & Grootendorst, *Hands-On Large Language Models*
- `bk*`/`b1*`/`b2*` (29): Burkov, *Hundred-Page Language Models Book*
  (2025) — b101 = Ch 1 Eq 1.1 p.20 style
- `at*` (2): Vaswani et al. (2017)
- Plus 18 arXiv papers per books.csv for specific modules.

## Method (inherits the standing batch discipline)

PDF is truth; verify every chapter/eq/page citation before use.
Implement the stated formula natively (numpy only, no torch/sklearn
deps); test with assertions a stub cannot pass; anchors generated from
runs; collision-scan R names before mirroring; three-way parity;
`run_fn_subset.sh` for any multi-file tests/fn run; full sweep on l14
before declaring a cluster done.

Scope note (Vee 2026-07-27, causalbert precedent): optional pretrained
weights from proper sources are allowed as an OPTIONAL utility; the
native path must exist and be the default. Modules whose formula line
is operationally untestable ("load checkpoint; fine-tune", e.g. hmtvp)
get a real native mini-implementation of the concept or an honest
NotImplementedError with a written reason — never a mean-of-inputs.

## Clusters (planned batches, smallest coherent first)

- [x] W1 Burkov (29): COMPLETE 2026-07-28. All 29 modules real
  (b101-b111 Ch 1 linear/logistic/BCE with closed-form gradients,
  b201-b203 Ch 2 CE + next-token MLE, bk* n-gram smoothing family,
  TF-IDF, repetition penalty, weight tying, Elman step, and a real
  reverse-mode autodiff for bkcgr). Eq 1.1/1.2 PDF-verified at
  printed p. 20/22. 30 cluster tests + 29 re-fixtured legacy files =
  88 green; gradients pinned to central finite differences. R parity
  `burkov_lm_native.R` (28 exports), 87 assertions on l14, mirrored
  byte-identical to rmorie.
- [x] W2 Vaswani (2) + Alammar (53): COMPLETE 2026-07-29. All 55
  real: full attention family (SDP/MHA/GQA/MQA/sliding-window/KV
  cache, the cache step proven equal to full attention recomputed),
  heads + embeddings, the contrastive loss family, retrieval metrics,
  greedy/sampled decoding on the shared LCG, text/RAG utilities,
  HDBSCAN-style clustering, UMAP objective descent, LDA collapsed
  Gibbs, BERTopic, a trainable softmax head, NSW ANN measuring its
  own accuracy, and 13 orchestration modules around caller-supplied
  models. 42 cluster tests + 55 re-fixtured legacy = 240 Python green;
  R port `alammar_llm_native.R` (55 exports), 141 parity assertions
  on l14 green FIRST RUN -- the LCG sampler, LDA chain and TSDAE
  deletions reproduce Python token for token.
- [ ] W3 Kamath (243): by chapter, eq-numbered — the most spec-like.
  Chapter census (km modules citing "Kamath et al (2024), Ch N"):
  Ch2 41 (encoders/attention basics), Ch3 12, Ch4 11, Ch5 12, Ch6 33,
  Ch7 3, Ch8 16, Ch9 22, unparsed-header 93 (section-style citations —
  re-grep with looser pattern when starting). Verify the Kamath page
  map against the PDF before citing; km001 = Ch 2, Eq 2.1, p. 30.
  Suggested tranches: Ch2 first (overlaps the W2 attention core —
  collision-scan against attsdp/attmh family before writing), then
  Ch6, Ch9, Ch8, the small chapters, then the 93 unparsed.
  **Tranche 1 (Ch2, 41): Python DONE 2026-07-29** — km001-km041 real,
  Eq 2.12 delegates to the shared attsdp core; Eq 2.19's
  mask-INSIDE-the-scaling convention pinned against Vaswani's (finite
  masks split, -inf agrees); the Eq 2.20-2.33 loss family shares one
  validated -mean-log-P core; MoE gate sparsity exact. 35 cluster
  tests + 82 re-fixtured legacy = 107 green. R parity DONE: kamath_ch2_native.R (26 exports, Eq 2.12/2.15 delegating to the alammar core), 59 assertions green on the first l14 run, mirrored to rmorie.
- [~] W4 Géron hm* (329): **tranche 1 (40) Python DONE 2026-07-29**
  via the Opus-draft + lead-verify pipeline. Real implementations
  incl. a reverse-mode autodiff tape, full MLP backprop, Mann-Whitney
  AUC, bit-exact bfloat16 rounding, AdaBoost stumps, LCG-bootstrap
  bagging with OOB, and forward passes for the BERT family with
  LCG-deterministic weights. 102 agent tests + 247 doctests, all
  re-verified by the lead against the installed repo tree; lead's
  independent probes: AlexNet param count 62,378,344 vs hand
  arithmetic, AUC vs brute-force pair counting. 289 remain.
- [~] W5 Géron gr* (201): **tranche 1 (40) Python DONE 2026-07-29**,
  same pipeline. Real: Adam/AdamW/Adamax with bias correction pinned
  at t=1 (a version without correction is off by 100x and a test
  asserts it), 1cycle and other schedules as full curves, BPTT with
  the hand-unrolled convention documented, a pure-numpy
  Jonker-Volgenant Hungarian verified against exhaustive permutation
  search, deterministic BPE, DCGAN/DDIM/DDQN steps. 94 agent tests +
  231 doctests re-verified; lead probes: Hungarian vs brute force on 3
  LCG matrices, BPE determinism, Adam first-step size == eta.
  161 remain.
- **R parity for W4/W5 tranche 1: PENDING** (the Kosorok lesson says
  do it per cluster — next session's first job).
- [ ] R parity per cluster, not at the end (Kosorok lesson).

## Repo-wide residue (census 2026-07-28, post-W1)

36,491 fn modules. ~16,170 still carry the wrapped template:
11,121 auto-extracted (long book filenames; triaged in
AUTOEXTRACTED_TRIAGE.md but NOT implemented) + ~5,050 hand-named.
Hand-named split: 828 DL shelf (this worklist) + **4,223 hand-named
stubs outside any worklist** (a2cv, abcgp, abcrej, adamopt, ...) citing
other books. Those 4,223 need their own shelf census after the DL
waves.

## Findings

- The wrapped template evaded the placeholder census: albow claims
  bag-of-words, returns the MEAN of the token array cast to float.
  Same body in 857 DL modules and (shared shape) the 16k autogen stubs.
- W1: the repetition-penalty invariant is the ODDS against unpenalised
  tokens, not absolute probability — with several tokens penalised at
  once, softmax renormalisation can RAISE one penalised token's
  probability when another falls further (measured: token 1 gained
  0.0003 while token 0 lost). First test asserted the naive claim and
  failed against correct code.
- W1 legacy tests passed 100-length arrays where scalars belong
  (w as a vector into float(w)); all 29 re-fixtured with worked
  values and refusal cases.
- W2: Prim's MST in the HDBSCAN core needs an explicit visited mask --
  np.minimum resurrects in-tree nodes and every edge connects the same
  two points; caught on planted blobs. A single largest-gap flat cut
  merges blobs when one far outlier dominates the gap ladder; replaced
  with the cluster-count-maximising cut. Repetition-penalty-style test
  traps again: an angular "disconnected" ANN entry that points at node
  0 IS connected; a min_samples=2 core distance on 2-point blobs
  reaches across blobs.


## Status correction 2026-07-29

The W4d commit message (541775f0a7) claims "ALL 857 DL placeholders
now real" -- WRONG at commit time. W4a (73 hm* modules) was still
drafting; its agent stalled on API errors three times. True count at
that commit: 784/857 real. Census cross-check: 4,296 hand-named
wrapped-template stubs remained = 4,223 non-DL + W4a's 73. This note
supersedes the commit message; the closing commit for W4a is the one
that may claim completion.
