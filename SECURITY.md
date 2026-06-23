# Security policy & architecture

`morie` (Python + R + libmorie C++ core) is the **open scientific core**
of the morie toolkit. AGPL-3.0-or-later. Distributed through PyPI,
r-universe, Homebrew, GHCR, and (eventually) CRAN.

This is research software: it analyses Canadian police, corrections, and
oversight data drawn entirely from **public open-data portals** (TPS Hub,
data.ontario.ca, opendata.toronto.ca, Chicago / NYC / Vancouver, Statistics
Canada CODR). It does not warehouse private data. It does ship cryptographic
primitives (libsodium + liboqs PQC) used by the `morie.crypto` subpackage
for unrelated downstream applications.

## Reporting a vulnerability

Email **<vsruhela@proton.me>** — **do not** open a public GitHub Issue
for security reports. GitHub's [private vulnerability reporting](https://github.com/rootcoder007/morie/security/advisories/new)
is also enabled. PGP preferred:

```
gpg --recv-keys F2A44D5982E7585E48DF861E335990B9336F7DD6
```

Please include:

- Description + impact + CVSS estimate.
- Reproducer (script, dataset slice, expected vs actual).
- `morie.__version__` / `packageVersion("morie")` and platform.
- Whether you want CHANGELOG credit.

**SLA**

| Severity                                                    | Acknowledge | Fix or mitigation |
| ----------------------------------------------------------- | ----------- | ----------------- |
| High (RCE, crypto break, code injection via API)            | 72 hours    | 14 days           |
| Moderate (DoS, integrity bug in numeric kernel)             | 72 hours    | 30 days           |
| Low (info leak, misleading docstring with security impact)  | 72 hours    | 90 days           |

No bug bounty (yet). Valid reports get CHANGELOG + release-notes credit
by default.

## Threat model

morie is a **library** consumed by data scientists, statisticians, and
oversight researchers. The host machine and user are trusted; remote
data sources are not.

**Adversaries we model:**

1. **Hostile open-data response.** A poisoned CKAN / Socrata / ArcGIS
   reply tries to land malicious bytes via the `morie.ingest_*` and
   `morie.tps_*` / `morie.otis*` loaders. Mitigations live in
   [`src/morie/ingest/`](src/morie/ingest/) and the shared
   `morie_http` C++ client at [`libmorie/`](libmorie/):
   - HTTPS only; cert verification on by default.
   - Streaming downloads with per-request `max_bytes` ceiling.
   - JSON parsed by `nlohmann::json` (no `eval`, no R `parse()` on
     remote strings).
   - Per-portal token-bucket throttle (4 req/s default).
   - No remote string is ever passed to `system()` / `subprocess`
     / `eval(parse(...))` on either Python or R side.

2. **Malicious dependency.** A compromised PyPI / CRAN / r-universe
   transitive dep. Mitigated by Dependabot, OSV-Scanner, CodeQL,
   pinned wheels in [`pyproject.toml`](pyproject.toml), and SLSA
   build provenance attached to every release.

3. **Numeric-kernel correctness as a security property.** Wrong
   statistical results in an oversight context have real consequences
   (false-positive or false-negative claims about police use of force,
   prison segregation, etc.). The "Mandela" misclassification rate
   work treats methodological bugs as security-class bugs: we ship
   Py↔R parity tests, golden-fixture pinning, and per-module
   correctness audits documented in `audit/`.

4. **Compromise of the crypto subpackage.** `morie.crypto` and
   `morie_crypto_sym.cpp` / `morie_crypto_pqc.cpp` ship libsodium +
   liboqs primitives. A bug here is downstream-fatal. Mitigated by
   delegating to upstream-vetted libraries (no rolled-our-own primitives)
   and by parity tests against test vectors.

5. **AGPL §13 network use.** morie can be deployed behind a network
   service. The AGPL obligates source-availability. Not a "security"
   threat in the usual sense but is a license-compliance one — we
   surface this in [`LICENSING.md`](LICENSING.md) and [`NOTICE`](NOTICE).

**Assets we protect:**

- **Numeric correctness** of the public-facing estimators.
- **Cryptographic correctness** of `morie.crypto`.
- **User trust** that `pip install morie` doesn't shell out to a
  network, exfiltrate data, or write to surprising paths.
- **CRAN policy compliance** (no writes to user `$HOME` by default;
  tempdir-only unless the user opts in via `R_user_dir()` —
  documented after the v0.9.4 archive incident).

**Out of scope:**

- **Host-OS compromise.** If your Python / R / OS is owned, we can't
  help.
- **Confidentiality of analyses.** Outputs live in the user's R/Python
  session; we don't manage them.
- **DoS against open-data portals.** We rate-limit ourselves
  politely; we don't try to detect or defend against portal-side
  DoS targeting us.
- **AI-side-channels.** `morie.llm` shells out to Ollama / Gemini /
  Vertex with user-supplied keys; we don't pretend to defend against
  prompt-injection in the user's own data.

**Trust boundaries:**

| Boundary                              | Crossing                                            |
| ------------------------------------- | --------------------------------------------------- |
| Network (open-data) → Python session  | `morie.ingest_*`, `morie_http.cpp`                  |
| Network → R session                   | `R/ingest_*.R` + libcurl via `morie_http`           |
| Crypto API → user secret              | `morie.crypto` / `src/morie_crypto_*.cpp`           |
| Python ↔ R parity surface             | `morie.fn` registry + RcppExports                   |
| User key (LLM API) → external endpoint| `morie.llm` (user-supplied, never logged)           |

## Cryptographic posture

`morie.crypto` and the R-side mirrors (`src/morie_crypto_sym.cpp`,
`src/morie_crypto_pqc.cpp`) consolidate the project's classical and
post-quantum primitives. The same primitives back the rmorie-cli
Receipt verifier (downstream) and any future at-rest sealing of
research artifacts.

- **KEX:** X25519 (libsodium) + ML-KEM-768 (liboqs) — hybrid.
- **Signatures:** Ed25519 (libsodium) for in-process integrity;
  Ed25519 + ML-DSA-65 hybrid for any downstream license / receipt
  payload.
- **AEAD:** ChaCha20-Poly1305 (libsodium).
- **KDF:** Argon2id with sane params for OS-keychain-derived keys;
  HKDF-SHA-512 for static derivations.
- **Hash:** SHA-256 (audit chain), SHA-512 (KDF context).
- **Libraries:** [libsodium](https://doc.libsodium.org/) (classical
  primitives, memory zeroization, constant-time compare) and
  [liboqs](https://github.com/open-quantum-safe/liboqs) (PQC).
  C++ glue under [`src/morie_crypto_sym.cpp`](src/morie_crypto_sym.cpp)
  and [`src/morie_crypto_pqc.cpp`](src/morie_crypto_pqc.cpp) — Python
  bindings via nanobind, R bindings via Rcpp.

**Why PQC hybrid:** harvest-now-decrypt-later resistance for any
artifacts encrypted with morie.crypto (sealed data slices, downstream
license payloads). Pure post-quantum is too new; classical-only is
unforward. Hybrid is the conservative middle.

## Control mapping

| Requirement                                | Where                                                                                                | ITSG-33     | NIST 800-53 (Mod) | OWASP ASVS L2 | Ontario MGCS IT Sec |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------- | ----------------- | ------------- | ------------------- |
| Software integrity verification            | Release artifacts GPG-signed; SHA-256 sidecars                                                       | SI-7        | SI-7              | V10.3.1       | §6.3 Integrity      |
| Cryptographic key management               | `morie.crypto` + `src/morie_crypto_*.cpp` (libsodium + liboqs)                                       | SC-12, SC-13| SC-12, SC-13      | V6.2, V6.4    | §6.7 Crypto         |
| Approved algorithms (PQC-future-proof)     | X25519+ML-KEM-768, Ed25519+ML-DSA-65, ChaCha20-Poly1305, Argon2id, SHA-256/512                       | SC-13       | SC-13             | V6.2.5        | §6.7 Crypto         |
| Input validation (remote JSON)             | `src/morie/ingest/*`, `libmorie/` HTTP client                                                        | SI-10       | SI-10             | V5.1.4        | §6.4 Network        |
| Polite-by-default HTTP (rate-limit)        | `morie_http` token-bucket, exponential backoff on 429/5xx                                            | SC-5        | SC-5              | V14.4.1       | §6.4 Network        |
| HTTPS + cert verification on by default    | `libmorie/morie_http.cpp` libcurl `CURLOPT_SSL_VERIFY*`                                              | SC-8, SC-13 | SC-8, SC-13       | V9.1, V9.2    | §6.4 Network        |
| No `eval` of remote content                | Code-review enforced; CodeQL Python+C++ rules                                                        | SI-10       | SI-10             | V5.2.4        | §6.2 Change ctrl    |
| Numeric-kernel correctness (Py↔R parity)   | `tests/` Py↔R parity suites; `audit/` per-module bug reviews                                         | SI-7        | SI-7              | V11.1.3       | §6.3 Integrity      |
| No writes outside `tempdir()` by default   | CRAN policy compliance — see `R/utils-cache.R` / Python `morie/_paths.py`                            | AC-3        | AC-3              | V12.3.6       | §5.2 Access         |
| Reproducible build                         | `pyproject.toml` (scikit-build-core, pinned deps), `DESCRIPTION` (pinned R deps + Remotes)           | CM-2        | CM-2              | V14.2.1       | §6.2 Change ctrl    |
| SBOM on release                            | [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml) + release-debrpm.yml  | SR-3        | SR-3              | V14.2.6       | §6.2 Change ctrl    |
| SLSA L3 provenance                         | `actions/attest-build-provenance` (release-debrpm.yml + docker-publish.yml)                          | SR-4        | SR-4              | V14.2.6       | §6.2 Change ctrl    |
| Static analysis on every push              | [`.github/workflows/codeql.yml`](.github/workflows/codeql.yml) (C++ + Python)                        | SA-11       | SA-11             | V1.1.4        | §6.2 Change ctrl    |
| R CMD check + coverage on every push       | [`.github/workflows/r-cmd-check.yml`](.github/workflows/r-cmd-check.yml), `r-coverage-and-lint.yml`  | CM-4        | CM-4              | V1.14.1       | §6.2 Change ctrl    |
| Version-drift guard                        | [`.github/workflows/version-drift.yml`](.github/workflows/version-drift.yml)                         | CM-2        | CM-2              | V14.2.1       | §6.2 Change ctrl    |
| AGPL-3.0 compliance (network use)          | [`LICENSING.md`](LICENSING.md), [`NOTICE`](NOTICE)                                                   | SA-4        | SA-4              | V1.1.1        | §3 Acceptable use   |
| Vulnerability disclosure                   | This document + GitHub Security Advisories                                                           | IR-6        | IR-6              | V1.1.4        | §7 Incident         |

ITSG-33: Treasury Board of Canada IT Security Guidance. NIST 800-53 Rev 5 moderate baseline. [OWASP ASVS 4.0.3](https://github.com/OWASP/ASVS/).

## Supply chain

- **Reproducible builds.** scikit-build-core for the Python wheel
  (CMake-driven libmorie). All transitive deps pinned in
  [`pyproject.toml`](pyproject.toml) and [`DEPENDENCIES.csv`](DEPENDENCIES.csv).
  R deps pinned via `DESCRIPTION` `Depends` + `Remotes`.
- **SBOM.** CycloneDX via Syft attached to every Docker / deb / rpm
  release. PyPI wheel ships a CycloneDX sidecar.
- **Build provenance.** SLSA L3 via
  [`actions/attest-build-provenance`](https://github.com/actions/attest-build-provenance);
  verify with `gh attestation verify`.
- **Dependency scanning.** Dependabot + OSV-Scanner + GitHub
  Dependency-Review on PRs.
- **Static analysis.** CodeQL on every push + scheduled weekly.
- **Signed releases.** GPG-signed Git tags
  (`F2A44D5982E7585E48DF861E335990B9336F7DD6`); SHA-256 sidecars on
  every release artifact.
- **CI action pinning.** All `uses:` lines in `.github/workflows/`
  pinned by full commit SHA, not tag.

## Audit & non-repudiation

- **Per-release transcript** — GitHub Release artifacts, the build
  attestation, and the SBOM together reconstruct what shipped from
  what source.
- **Versioning discipline** — version-drift guard (`.github/workflows/
  version-drift.yml`) blocks releases where `VERSION` / `pyproject` /
  `DESCRIPTION` / Zenodo metadata disagree.
- **Issue audit chain** (roadmap v1.0) — SHA-256 hash-chained
  per-Mandela-finding registry so downstream papers / press citing
  morie's outputs reference a verifiable artifact, not a transient
  number.
- **RFC 3161 timestamping** (roadmap v1.0) over the release transcript
  from a Canadian TSA (e.g. `timestamp.entrust.net`).

## What this component does NOT defend against

- **A user who runs `morie` on a compromised host.** All bets off.
- **Malicious R / Python *packages other than morie itself*** installed
  in the same session. We can't sandbox sibling packages.
- **Hostile open-data portals serving wrong-but-well-formed data.**
  We can detect shape changes; we can't detect *intentional* false data
  from a trusted upstream.
- **CRAN / PyPI / r-universe mirror compromise.** Mitigated indirectly
  via GPG-signed tags + SHA-pinned provenance, but a fully-end-to-end
  reproducible install verification is the user's responsibility.
- **Side-channels in libsodium / liboqs / numpy / Rcpp.** We use
  upstream; we don't backport security patches independently.
- **The user choosing to feed sensitive data to `morie.llm` Vertex /
  Gemini / Claude endpoints.** That's a user policy decision; the
  library never logs prompts.
- **Pre-alpha API stability.** v0.x APIs may shift. Don't bet
  production infrastructure on a pre-alpha estimator.

## Roadmap

**Wave 1 — shipped (v0.9.5.x)**

- libsodium + liboqs C++ port for the crypto subpackage.
- 352 `morie_*`-prefixed exports (collision-free).
- Py↔R parity test suite (60+ BOTH-language modules covered).
- CodeQL + Dependabot + r-cmd-check + coverage CI.
- Polite token-bucket HTTP client in `libmorie`.
- AGPL-3.0-or-later relicense + NOTICE updates.
- CRAN-policy compliance retrofit (no surprise `$HOME` writes).

**Wave 2 — next (v1.0.0 alpha)**

- Sigstore cosign for Python wheels + Docker images.
- macOS / Windows Authenticode-signed binary artifacts.
- RFC 3161 timestamping over release transcripts.
- Hash-chained per-finding registry for the empirical paper outputs.

**Future**

- CRAN submission (currently on hold per Uwe Ligges' archive guidance;
  channel via r-universe is healthy).
- AGPL §13 — Source-availability network endpoint for any hosted
  morie service.
- Side-channel-resistant builds of `morie.crypto` (constant-time
  rebuilds verified via dudect / ctgrind).

---

Maintainer: **Vansh Singh Ruhela** ([rootcoder007](https://github.com/rootcoder007))
&nbsp;·&nbsp; <vsruhela@proton.me>
