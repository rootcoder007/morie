# Template-B placeholders — 333 modules

A **second** placeholder template, distinct from the
`result = np.mean(...)` / `se = np.std(...)` form the earlier census
counted. Found by chasing pytest warnings during the Kosorok shelf.

```python
estimate = np.median(<first arg>)
se = 1.2533 * np.std(<first arg>, ddof=1) / np.sqrt(n)
```

Two problems, both silent:

1. **NaN standard errors.** When the first argument is a scalar,
   `np.std(..., ddof=1)` divides by zero degrees of freedom and `se`
   is NaN with only a RuntimeWarning to show for it.
2. **Every other argument is ignored.** `ksr064` documented the Cox
   partial likelihood and took `Z`, `V`, `d` — then computed the
   median of `beta`.

Census: **333 modules**, **zero overlap** with the 16,182 template-A
placeholders.

## Families

| Book | Count |
|---|---:|
| Rangayyan | 37 |
| Horowitz | 47 |
| Fauzi | 22 |
| Kosorok | 15 |
| Wasserman | 7 |
| Molak | 11 |
| Cochran | 5 |
| Schabenberger | 4 |
| (tail: Shrout & Fleiss, Robins-Rotnitzky, Géron, Efron & Tibshirani, Burkov, Yohai, …) | rest |

## Progress

- [x] **ksr064, ksr069** — real Cox partial likelihood (eq. 3.4) and
      Breslow baseline, SEs from the observed information.
- [x] **Rangayyan basic + filters + adaptive (15)** — rng007–010,
      rgmavg, rng039, rng087, rng097, rng137, rng140, rng156, rng159,
      rng165, rng166, rng194. 7/7 tests green.
- [x] **Rangayyan spectral (7)** — rgacf, rgperio, rgwelch, rgyw,
      rgarsp, rgpsdacf, rgbwbnd. 13/13 tests green.
- [x] **Rangayyan biomedical (15)** — rgburg, rgcepsp, rgeegsp,
      rgelast, rgenvgm, rgpdfest, rgrmsnw, rgtfe, rgtwamx, rng017–020,
      rng190, rng211. 24/24 tests green. **Rangayyan complete: 37/37.**
- [x] **R parity for Rangayyan (37)** — `R/rangayyan_native2.R`,
      13 exports, 39 tests green, mirrored byte-identical into both
      trees. Collision scan skipped the five already-covered
      `morie_dsp_*` functions.
- [ ] Horowitz (47), Fauzi (22), Kosorok remainder (13), tail (~272)
      — **R parity in the SAME commit as the Python from here on**

## Distinctions the repairs preserve

Each module states the trade-off its formula embodies rather than
presenting one convention as neutral:

- `rgacf` returns both the unbiased (divisor N − |m|) and biased
  (divisor N) autocorrelations. The unbiased one is *not* guaranteed
  positive semi-definite, so it can produce an unstable AR fit.
- `rgyw` therefore feeds the **biased** ACF to the Toeplitz solve on
  purpose, and reports whether the fitted model is stable.
- `rgperio` says plainly that the periodogram is inconsistent — more
  samples buy resolution, not precision — which is why `rgwelch`
  exists.
- `rgwelch` carries the window-power normalisation U explicitly;
  without it a Hann window biases the PSD low by about 2.7×.
- `rng008` returns mean square *and* variance, which coincide only for
  a zero-mean signal.
- `rng010` uses the book's divisor N and returns the N − 1 form
  alongside.
- `rng097` reports its 3.5-sample delay: an even-length boxcar cannot
  be delay-corrected by an integer shift.
- `rgtfe` refuses a single segment: coherence from one segment is
  identically 1 at every frequency and certifies nothing.
- `rgtwamx` truncates to an EVEN beat count, since 0.5 cycles/beat is
  only an exact FFT bin when the beat count is even.
- `rgenvgm` requires R peaks rather than guessing alignment —
  averaging unaligned beats smears the envelope it is meant to show.
- `rgelast` returns spectral descriptors and a `calibrated: False`
  flag instead of inventing an absolute stiffness value; the
  relationship is monotone but the calibration is subject-specific.
- `rgeegsp` refuses fs ≤ 60 Hz, which cannot represent the 13–30 Hz
  beta band it would otherwise report a number for.

## Parity rule, restated after a lapse

Kosorok and the Rangayyan template-B work were both completed
Python-only and only mirrored afterwards, on being challenged. That
broke the standing rule. From Horowitz onward the R mirror ships in
the **same commit** as the Python, never as a follow-up.

Two parity bugs the cross-language anchors caught, neither visible
from the Python side alone:

- `morie_functional_delta` used the raw Jacobian where the delta
  method needs it applied to the deviation; the remainder came out
  −395.99 instead of 0.01.
- `morie_moving_average` filled `stats::filter`'s NA startup transient
  by dividing by the number of available samples. Python's zero-padded
  convolution divides by M throughout, so `y[2]` on `0:9` with M = 4
  is 0.25, not 0.5.
