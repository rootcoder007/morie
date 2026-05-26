# MORIE 森 — The Fates' Forest

**Multi-domain Open Research and Inferential Estimation**

<sub>*Pronounced /ˈmɔɪraɪ ˈmoɾi/ — "MOY-rye MOH-ree". A compound name: **Moirai** (Μοῖραι), the three Greek Fates, paired with **森** (mori), Japanese for "forest" — three trees (木) stacked into one ideogram. A forest of methods named after the Fates.*</sub>

A multi-domain scientific computing toolkit (Python and R) for observational inference, with sociolegal, signal-processing, cryptographic, spatial-statistics, statistical-physics, and psychometrics modules. Hosts the MRM framework as a primary application for Canadian carceral, police, and oversight data analysis.

[![R CMD check](https://github.com/rootcoder007/morie/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/rootcoder007/morie/actions/workflows/r-cmd-check.yml)
[![CI](https://github.com/rootcoder007/morie/actions/workflows/ci.yml/badge.svg)](https://github.com/rootcoder007/morie/actions/workflows/ci.yml)
[![CodeQL](https://github.com/rootcoder007/morie/actions/workflows/codeql.yml/badge.svg)](https://github.com/rootcoder007/morie/actions/workflows/codeql.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-a42e2b.svg)](https://github.com/rootcoder007/morie/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/morie.svg)](https://pypi.org/project/morie/)
[![r-universe](https://img.shields.io/badge/r--universe-rootcoder007-276DC3)](https://rootcoder007.r-universe.dev/morie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> ⚠️ **Pre-alpha (v0.x).** MORIE is in pre-alpha. The first alpha milestone is **v1.0.0**; everything before that is point-releases of pre-alpha code. APIs may shift, datasets may move, and findings may be refined between minor versions. Paper sources are at [`papers/`](https://github.com/rootcoder007/morie/tree/main/papers) (LaTeX).

## Installation

> Full step-by-step install guide with platform-specific notes (PEP 668 on Debian, python 3.13 segfault on Raspberry Pi OS, etc.) is at **[INSTALLATION.md](https://github.com/rootcoder007/morie/blob/main/INSTALLATION.md)**.

morie is a Python (and R) package — once Python is present it is `pip install morie`. If you are starting with **nothing installed**, INSTALLATION.md opens with **[Step 1 — install the prerequisites](https://github.com/rootcoder007/morie/blob/main/INSTALLATION.md#step-1--install-the-prerequisites)**: every tool you might need (Python, `curl`, `bash`/WSL, Git Bash, `winget`, Homebrew, Docker, R) with its official download. The short version:

- **Windows** — install Python from [python.org](https://www.python.org/downloads/) (on the first screen tick **Add python.exe to PATH**), then `pip install morie`. Full walkthrough: [Windows](#recommended--windows) below. Windows has no `curl`/`bash`, so the one-liner does not apply there.
- **macOS / Linux** — the one-liner below sets up everything. It needs `curl` and `bash`, which macOS has built in and most Linux ships.
- **Already have Python ≥3.10** — just `pip install morie`.

### For terminal users — one-liner (Linux / macOS / WSL)

The simplest path **if you have a terminal with `curl` and `bash`** — both are built into macOS and preinstalled on most Linux (**Windows has no `bash`**, so use the installer above instead). It then bootstraps everything else for you: Python via `uv`, a managed venv, and the morie wheel. No pre-existing Python or `pip` needed.

```bash
curl -fsSL https://rootcoder007.github.io/morie/install.sh | bash
```

Or, with R alongside Python:

```bash
curl -fsSL https://rootcoder007.github.io/morie/install.sh | bash -s -- --auto
```

After install, `~/.local/bin/morie` is a thin shim into the managed venv at `~/.venvs/morie`. Full install instructions, channel comparison, and platform-specific notes are at **[rootcoder007.github.io/morie/#quick-start](https://rootcoder007.github.io/morie/#quick-start)**.

> On minimal Linux containers (Alpine, slim Debian) that ship without `curl`, install it first: `apt-get install -y curl` or `apk add curl`. macOS already has `curl` built in.

### Recommended — Windows

Windows doesn't ship `curl`, `bash`, `python`, or `R`, so the Linux/macOS one-liner above won't run there. The path that works on **any** Windows with no prerequisites:

1. Install Python from **[python.org/downloads](https://www.python.org/downloads/)** — on the first installer screen, **tick "Add python.exe to PATH"** (skipping this is the No. 1 cause of `python` being "not recognized" in the terminal).
2. *(Optional — for the R package)* install R from **[cran.r-project.org/bin/windows/base](https://cran.r-project.org/bin/windows/base/)**.
3. Open **PowerShell** and install morie:

```powershell
python -m pip install --upgrade pip
python -m pip install morie
python -c "import morie; print(morie.__version__)"
```

For the R package: `Rscript -e "install.packages('morie', repos=c('https://rootcoder007.r-universe.dev','https://cloud.r-project.org'))"`

Prefer a package manager? If `winget --version` works on your machine, `winget install -e --id Python.Python.3.12` (and `RProject.R`) installs the prerequisites in one line each — but `winget` is absent from many Windows installs, so the installer steps above are the reliable default. The full Windows walkthrough, including fixes for common errors (`python` opening the Microsoft Store, PowerShell execution policy, long-path), is in **[INSTALLATION.md](https://github.com/rootcoder007/morie/blob/main/INSTALLATION.md)**.

### Python — Homebrew (macOS / Linuxbrew)

If you don't have Homebrew yet, install it first (macOS ships `curl` and `bash`, so this works out of the box):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```bash
brew tap rootcoder007/morie
brew install morie
```

The tap repo is [`rootcoder007/homebrew-morie`](https://github.com/rootcoder007/homebrew-morie). It pulls morie's source distribution from PyPI and bundles a self-contained `python@3.12` venv — no system Python required.

### Python — PyPI (manual; requires `pip` already installed)

```bash
pip install morie
```

> **Heads-up:** modern Debian / Ubuntu / Raspberry Pi OS forbid `pip` outside virtual environments (PEP 668), and the system `python3` on Raspberry Pi OS 13 segfaults on importing the SciPy stack. If `pip install morie` errors or `import morie` segfaults, use the one-liner above instead — it handles both cases automatically.

### Python — Docker (no local dependencies)

```bash
# Latest stable
docker run --rm ghcr.io/rootcoder007/morie:latest morie --help

# Pin to a specific version (recommended for reproducibility)
docker run --rm ghcr.io/rootcoder007/morie:0.9.5.4 morie --help
```

Multi-arch image published on every release with both versioned and `:latest` tags. Requires only Docker — no Python, no pip.

### R — CRAN (when available) or r-universe

```r
# Stable from CRAN (when listing is live)
install.packages("morie")

# Nightly binary builds (recommended while CRAN listing is rolling out)
install.packages(
  "morie",
  repos = c(
    rootcoder007 = "https://rootcoder007.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
```

## Quick start

```python
import morie

# Load a built-in dataset
df = morie.load_dataset("otis-2025")

# Run an MRM module on OTIS data
from morie.otis_all_analyze import analyze_a01_mrm
result = analyze_a01_mrm(df)
print(result)
```

## What's new

Per-release user-facing changes are now in [WHATS_NEW.md](WHATS_NEW.md).
For the R-package changelog see [r-package/morie/NEWS.md](r-package/morie/NEWS.md).
For the planned roadmap see [ROADMAP.md](ROADMAP.md).

## Documentation

Full documentation is at [rootcoder007.github.io/morie](https://rootcoder007.github.io/morie/).

## Citation

If you use morie in your research, please cite **both software papers** (R and Python)
and, where applicable, the **MRM framework paper**, the **Hawkes methodology paper**,
and the **empirical applications paper**.

```
# Software paper — R
Ruhela, V. S. (2026). morie: Multi-domain Open Research and Inferential
Estimation in R (v0.9.5.12).
https://github.com/rootcoder007/morie

# Software paper — Python
Ruhela, V. S. (2026). morie: Multi-domain Open Research and Inferential
Estimation in Python (v0.9.5.12).

# MRM framework paper (theoretical foundations)
Ruhela, V. S. (2026). MRM Framework: Multi-Source Statistical Foundation
for Canadian Carceral, Police, and Oversight Data (v1).

# Hawkes-process methodology paper
Ruhela, V. S. (2026). Criminological Hawkes Process via MORIE: Markovian
and Non-Markovian Self-Exciting Point Processes for Toronto Crime (v1).

# Empirical applications paper
Ruhela, V. S. (2026). Solitary Confinement, Self-Excitation, and
Institutional Churn: Empirical Applications of MRM to Canadian Carceral
and Police Data (v1).
```

See [`CITATION.cff`](https://github.com/rootcoder007/morie/blob/main/CITATION.cff) for machine-readable citation metadata.


## Acknowledgments

### AI assistance

MORIE was developed with substantial assistance from frontier AI
assistants. The author retains full responsibility for the code, the
methods, and the scientific claims; AI assistance accelerated
implementation but does not change the attribution of the work.

- **Claude — Anthropic.** Anthropic's Claude family (Opus, Sonnet, and
  Haiku across the 4.x generation) was used extensively throughout
  development for code generation, refactoring, documentation, code
  review, and design discussions. Use was supported by Anthropic
  research-credit programs.

- **Gemini and Vertex AI — Google.** Google's Gemini 2.5 models (Pro and
  Flash) on the Vertex AI platform were used extensively for additional
  code generation, cross-checking Claude-generated code, multi-modal
  data analysis, and prototype evaluation. Use was supported by Google
  research-credit programs.

### Funding and infrastructure

- Anthropic — Claude API research credits.
- Google — Gemini / Vertex AI research credits.
- The author thanks **Glenn McNamara** — a 35-year career with the
  Ontario Government — for his methodological mentorship. He brings
  distribution theory, applied-statistics intuition for administrative
  data, and the judgment that grounds much of this framework. Glenn
  is the **M** in **MRM (Multilevel Reconciliation Methodology; people-credit reading: McNamara-Ruhela-Medina)** (catalyst).

- The author thanks **Prof. Angela Zorro Medina**, Centre for
  Criminology and Sociolegal Studies, University of Toronto, who is
  the author's **supervisor**, **methodological instructor**, the
  **domain-expert reviewer** of the preliminary methodological
  approach, and a **knowledge user** of the framework. The
  methodological lineage MRM follows is established in her work on
  anti-gang legislation (Zorro Medina, 2023, *The Effect of
  Anti-Gang Laws on Crime and Social Control*) — staggered
  two-way-fixed-effects identification, formal leads-and-lags
  Granger-causality diagnostics for parallel trends, multi-source
  data-integration over five jurisdictional sources, deterrence /
  routine-activities / certainty mechanism categorisation, and the
  inequality-effects-of-criminal-law framing — all of which
  directly shape MRM's empirical-statistical spine. Prof. Medina is
  the **M** in MRM (supervisor & reviewer).

### Data acknowledgments

Several MRM analyses use Statistics Canada and Health Canada Public
Use Microdata Files (PUMFs) — including the **Canadian Cannabis
Survey (CCS)**, the **Canadian Student Alcohol and Drugs Survey
(CSADS)**, the **Canadian Substance Use Survey (CSUS)**, the
**Canadian Alcohol and Drugs Survey (CADS, 2019;
[doi.org/10.25318/132500052021001-eng](https://doi.org/10.25318/132500052021001-eng))**,
and the **Canadian Postsecondary Education Alcohol and Drug Use
Survey (CPADS)** — along with Public Health Agency of Canada (PHAC)
and Canadian Institute for Health Information (CIHI) aggregates.
Although the analyses use Statistics Canada and Health Canada data,
the analyses, interpretations, and conclusions are those of the
author and do not represent the views of Statistics Canada or
Health Canada. Ontario open data (OTIS, A01-RCDD release; via
`data.ontario.ca`) and Toronto Police Service open data are used
under the same standard disclaimer.

## License

morie is licensed under the **GNU Affero General Public License, version 3.0 or later (`AGPL-3.0-or-later`)**, on both the Python and R sides. The AGPL is a strong copyleft license: anyone who distributes a modified morie — or offers a modified morie to users over a network — must publish their source. Modifications and improvements cannot be kept secret or taken closed-source.

- **Python and R packages** (`src/morie/`, `r-package/morie/`) — `AGPL-3.0-or-later`. See [`LICENSE`](https://github.com/rootcoder007/morie/blob/main/LICENSE).
- **Optional Linux kernel adjuncts** (`kernel-module/morie.c`, `daemon/morie_lsm.py`) — `GPL-2.0-only` (the Linux kernel ABI requires GPL for loaded modules). These are NOT part of the R / Python distribution; they are separately-licensed, independently-distributed adjuncts. See [`kernel-module/LICENSE-GPL2`](https://github.com/rootcoder007/morie/blob/main/kernel-module/LICENSE-GPL2).
- **Papers, data and documentation** — `CC BY-NC-SA 4.0` (Creative Commons Attribution-NonCommercial-ShareAlike) unless explicitly marked otherwise.

Full detail in [`LICENSING.md`](https://github.com/rootcoder007/morie/blob/main/LICENSING.md).

## Reporting issues / security

- General issues: [GitHub Issues](https://github.com/rootcoder007/morie/issues)
- Security vulnerabilities: see [`SECURITY.md`](https://github.com/rootcoder007/morie/blob/main/.github/SECURITY.md)
