<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# libmorie_core — the C ABI face of the morie numeric core

morie's numeric algorithms live **once**, in the header-only C++ core
(`../libmorie/morie_core.hpp`, `namespace morie::core`). This directory wraps
that core in a stable **`extern "C"` ABI** so any language that can call a C
function calls the *same compiled arithmetic* — no reimplementation, no
Python↔R↔CLI parity drift.

```
        libmorie/morie_core.hpp   (C++ core — the math, written once)
                   │
        capi/morie_c_api.{h,cpp}  (extern "C" façade — plain C types)
                   │
   ┌────────┬──────┴──────┬─────────┬──────────┐
  C/C++    Go (cgo)   Rust (FFI)  Python      R (.C)      WASM
                       rmorie-cli  ctypes     .Call       emscripten
```

## Build & test (standalone — does not touch the Python wheel build)

```sh
cmake -S capi -B capi/build
cmake --build capi/build
ctest --test-dir capi/build --output-on-failure
```

Produces `libmorie_core` (shared + static) + `morie_c_api.h`, plus a pure-C
test (`test_c_api.c`) that links the **shared** library and asserts the core's
numbers against hand-computed values.

## Current ABI surface (v0.1.0)

Summary stats (`morie_mean`, `morie_variance`, `morie_stddev`,
`morie_cor_pearson`, `morie_euclid_dist`), array kernels
(`morie_normal_pdf`, `morie_normal_logpdf`), special functions
(`morie_gamma_cdf_regularized`), and `morie_core_version()`.

**Extending:** add a thin forward in `morie_c_api.cpp` + a declaration in
`morie_c_api.h` for each additional `morie::core::` function (Hawkes
likelihoods, IPW weights, bootstrap). Never put math here — it belongs in the
core header. ABI rule: add functions, never change an existing signature.
