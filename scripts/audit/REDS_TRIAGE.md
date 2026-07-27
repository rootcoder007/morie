# The 138 reds: triage by cause

Measured 2026-07-27 on L14 (`~/fnrun/reds4.log`), full `tests/fn` sweep.

## Headline

**138 failing tests across 72 distinct modules. 66 of those 72 modules
already contain real, working implementations.** They fail because the
*generated test* hands them an input their contract rejects — not
because the implementation is wrong.

Only **4 placeholders + 2 missing modules** need implementation work.

This matters for planning. The unit of work is the **error signature**,
not the module: five families cover 129 of the 138.

## How the tests came to be wrong

The generated tests were written against the *placeholders*, which
returned a mean/se dict for any input whatsoever. Any fixture passed.
Once a module became a real implementation with real input validation,
the same fixture started failing.

`test_archm.py` is the archetype:

```python
def test_archm_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = arch_in_mean(x)          # ARCH-in-mean needs n >= 20
    assert "estimate" in result

def test_archm_edge():
    result = arch_in_mean(np.array([42.0]))
    assert result["n"] == 1           # asserts nothing about the model
```

`arch_in_mean` is a real ARCH(1)-in-mean fit via `scipy.optimize`. Five
observations cannot identify it, and the function says so. The test is
wrong; the code is right.

Note the second test also shows the deeper problem: `result["n"] == 1`
is true of *any* function that echoes its input length. It would pass
against a stub. Re-fixturing alone is not enough — the assertions have
to say something the implementation could get wrong.

## The five families

| Family | Modules | Failure | Fix |
|---|---|---|---|
| Series length | 17 | 5-point fixture to estimators needing n >= 20/30/2*period | Generate a series long enough to identify the model, with known parameters |
| Matrix shape | 11 | 1-D array where 2-D or square-symmetric is required | Build the right shape; for spatial weights, a real contiguity matrix |
| NN hyperparameter | 10 | arrays passed where `d_model`, `fan_in`, `kind` are scalars/strings | Pass the documented scalar; assert exact output shapes |
| Classification labels | 6 | continuous data where class labels are required | Discrete labels; assert against a hand-computable confusion matrix |
| Genuine defects | 9 | 7 AssertionErrors + `bnp_inference_suite`, `vecmF` missing | Investigate individually |

### Series length
`archm` `egrch` `tgrch` `regms` `kssup` `nbeat` `ucmod` `dccmd` `propc`
`johsn` `vecmf` `midas` `tarmd` `lilf` `volengle` `volcorpst` `dccgrch`

### Matrix shape
`cnn2d` `mxpol` `gearyc` `sarre` `sarla` `sptau` `cvxhl` `okrig`
`spblk` `mcnem` `kldivg`

### NN hyperparameter
`rotrp` `mhatf` `grpqa` `heinz` `posab` `vaenc` `ganls` `trfbl`
`tknbp` `tsnrd`

### Classification labels
`svmhg` `confm` `svmkr` `crba` `dtrsp` `kmnsc`

### Still placeholders (need implementation, not re-fixturing)
`dccgrch` (1137 b) `sgtadj` (915 b) `sgtnbe` (935 b) `studres` (751 b)

### Missing modules
`bnp_inference_suite` `vecmF` (note the capital F; `vecmf` exists)

## The standard for a rewritten test

Re-fixturing is necessary but not sufficient. Each rewritten test must
assert something a stub would fail:

- **Parameter recovery** over a known data-generating process, not
  merely that the output is finite. A wrapper that returns `x0` with
  `success=True` passes a finiteness check.
- **Rates across seeds**, not a single-seed point value, wherever the
  quantity is stochastic. Put the measured rate in the comment.
- **Absolute** tolerances where the target may be near zero —
  `pytest.approx(x, abs=...)`, since a relative tolerance on 0.3 is
  about one standard error at typical n.
- **Exact shapes and identities** for the deterministic families
  (attention, convolution, positional encoding), which admit
  hand-checkable answers.

## Progress

| Family | Status |
|---|---|
| NN hyperparameter | **done** -- all 10 incl. `trfbl` `tknbp` `tsnrd` |
| Matrix shape | **done** -- all 11 incl. spatial six, PDF-verified vs Schabenberger & Gotway |
| Series length | **done** -- all 16, DGP parameter-recovery tests |
| Classification labels | **done** -- all 6, ESL-anchored |
| Placeholders | **done** -- `sgtadj` `sgtnbe` implemented; `dccgrch` front-ends dccmd; `studres` was real |
| Missing modules | **done** -- `test_vecmF.py` deleted (stale duplicate); bnp suite test rewritten |
| Assertion reds | **done** -- logrnk/cgmth/studres/describe stale tests fixed; `morani` restored to the import table |

Real defects found and fixed (all masked by the placeholder-era tests):

- `mcnem`: 0.5 continuity correction where Edwards (1948) subtracts 1.
- `cvxhl`: `np.cross` on 2-vectors, removed in NumPy 2.0 -- raised on
  every call.
- `volengle`: docstring promised Engle's ARCH-LM; body was a KS
  normality template that ignored `q`. Reimplemented (Engle 1982
  Sec. 8).
- `volcorpst`: same KS template, ignored `horizons`. Reimplemented as
  a per-horizon distributional accuracy test (Corradi & Swanson 2006,
  J. Econometrics 135, 187-228 -- verified to exist).
- `archm`: L-BFGS-B returned its starting values verbatim on every
  input (delta = 0.0, alpha = 0.2 even on iid data). Powell + variance
  ceiling.
- `lilf`: reported the classical KS p-value as if it were Lilliefors,
  and described the direction of the error backwards. Now a Monte
  Carlo null.
- `kssup`: broke on current scipy (string dist + args maps "norm" to
  ndtr); now freezes the fitted distribution.
- `regms`: transition matrix orientation undocumented; now
  row-stochastic by contract.
- `sgtadj`/`sgtnbe`: mean-of-edge-array placeholders; real adjacency +
  Hashimoto matrices.
- `dccgrch`: spearmanr(X, X) placeholder, identically 1; now delegates
  to the real DCC engine.

Citation policy applied throughout: page-precise references verified
against the library PDFs (`data/datasets/userguides/other/pdf`), txt
used only as the search index. Schabenberger & Gotway eqs. (1.14),
(1.15), (5.13)-(5.16), (6.33)-(6.41) and ESL eq. (9.17) read in the
PDF itself; the Corradi-Swanson citation verified by web search.

## Final state (2026-07-27)

All 138 reds closed, plus the 18-module tail the five families did not
cover. Targeted re-runs of every previously failing file: green.
Cumulative real defects found and fixed under the placeholder-era
tests: mcnem, cvxhl, volengle, volcorpst, archm, lilf, kssup, regms,
sgtadj, sgtnbe, dccgrch, rng223, semthe, msm164, and the hrzc1 CLAD
intercept bug (slope 2.73-2.85 measured everywhere truth was 2.0).

R parity: reds_native.R (6 mirrors + archm ceiling) byte-identical in
rmorie and r-package/morie, 51/51 tests green in both; R McNemar and
Lilliefors verified already-correct, no mirror needed.

Placeholder population underway: see CAUSAL_PLACEHOLDERS.md.

## Reproducing the triage

```sh
# on L14
cd ~/morie-fnaudit
./.venv-fn/bin/python -m pytest tests/fn --no-header -q -p no:randomly \
    --timeout=120 --tb=line > ~/fnrun/reds.log 2>&1

grep '^FAILED' ~/fnrun/reds.log \
  | sed -E 's|^FAILED tests/fn/test_([a-zA-Z0-9_]+)\.py.*|\1|' | sort -u
```
