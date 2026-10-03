# MORIE 森 <img src="https://raw.githubusercontent.com/rootcoder007/morie/main/r-package/morie/man/figures/logo.png" align="right" height="139" alt="morie hex logo" />

**Multi-domain Open Research and Inferential Estimation.**

A multi-domain scientific computing toolkit (Python and R) for observational inference, with sociolegal, signal-processing, cryptographic, spatial-statistics, statistical-physics, and psychometrics modules. Hosts the MRM framework as a primary application for Canadian carceral, police, and oversight data analysis.

The R package ships **native causal-inference engines** — matching (nearest/Mahalanobis/exact/CEM/optimal/genetic/cardinality), double machine learning with clustered SEs, R-learner causal forests, T/S/X/DR meta-learners, design-based GLM, and a causal DAG toolkit (`morie_dag*`) — implemented in-package and cross-validated against MatchIt, DoubleML, grf, and dagitty rather than depending on them.

[![CI](https://github.com/rootcoder007/morie/actions/workflows/build.yml/badge.svg)](https://github.com/rootcoder007/morie/actions/workflows/build.yml)
[![CodeQL](https://github.com/rootcoder007/morie/actions/workflows/codeql.yml/badge.svg)](https://github.com/rootcoder007/morie/actions/workflows/codeql.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-a42e2b.svg)](https://github.com/rootcoder007/morie/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/morie.svg)](https://pypi.org/project/morie/)
[![rmorie on r-universe](https://rootcoder007.r-universe.dev/badges/rmorie)](https://rootcoder007.r-universe.dev/rmorie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Website](https://img.shields.io/badge/website-rmorie.com-1d1d1f.svg)](https://rmorie.com) [![Hosted LLM](https://img.shields.io/badge/hosted%20LLM-llm.rmorie.com-0066cc.svg)](https://llm.rmorie.com)

> The `morie` command line checks PyPI once a day for a newer release (fail-silent,
> cached); `import morie` makes no network request. Set `MORIE_NO_UPDATE_CHECK=1` to disable it.

## Installation

> Full step-by-step install guide with platform-specific notes (PEP 668 on Debian, python 3.13 segfault on Raspberry Pi OS, etc.) is at **[INSTALLATION.md](https://github.com/rootcoder007/morie/blob/main/INSTALLATION.md)**.

morie is a Python (and R) package — once Python is present it is `pip install morie`. If you are starting with **nothing installed**, INSTALLATION.md opens with **[Step 1 — install the prerequisites](https://github.com/rootcoder007/morie/blob/main/INSTALLATION.md#step-1--install-the-prerequisites)**: every tool you might need (Python, `curl`, `bash`/WSL, Git Bash, `winget`, Homebrew, Docker, R) with its official download. The short version:

> **Two steps, whatever the channel.** Install morie, then run
>
> ```bash
> morie interactive install
> ```
>
> once per user. The published package leaves out the five modules behind
> `morie repl`, `morie exec`, `morie agent` and `morie tui` (they run code you or a
> model type); that command fetches them for your installed version from the
> release tag on GitHub and checks each against the manifest inside the package.
> The one-liner installer below runs it for you. A verb that needs the layer says
> so and, on a terminal, offers to run it. Details: [the interactive layer](#the-interactive-layer--morie-repl-morie-exec-morie-agent-morie-tui).

- **Windows** — install Python from [python.org](https://www.python.org/downloads/) (on the first screen tick **Add python.exe to PATH**), then `pip install morie`. Full walkthrough: [Windows](#recommended--windows) below. Windows has no `curl`/`bash`, so the one-liner does not apply there.
- **macOS / Linux** — the one-liner below sets up everything. It needs `curl` and `bash`, which macOS has built in and most Linux ships.
- **Already have Python ≥3.10** — just `pip install morie` (on Debian, Ubuntu and Raspberry Pi OS use the venv three-liner below; their system `pip` refuses to install into `/usr`). The estimators take a pandas DataFrame, a CSV path, or a dict of columns; pandas is optional (morie ships its own frame core).

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

### Debian, Ubuntu, Raspberry Pi OS — plain `pip` in a venv

Debian-family systems mark the system Python "externally managed", so `pip install morie` stops with an error. A virtual environment is the supported way and needs nothing beyond the stock `python3` (add `sudo apt-get install -y python3-venv` if `venv` is missing):

```bash
python3 -m venv ~/.venvs/morie
source ~/.venvs/morie/bin/activate
pip install -U morie
morie --version
morie interactive install
```

Activate the venv (`source ~/.venvs/morie/bin/activate`) in each new shell, or call `~/.venvs/morie/bin/morie` directly.

### Recommended — Windows

Windows doesn't ship `curl`, `bash`, `python`, or `R`, so the Linux/macOS one-liner above won't run there. The path that works on **any** Windows with no prerequisites:

1. Install Python from **[python.org/downloads](https://www.python.org/downloads/)** — on the first installer screen, **tick "Add python.exe to PATH"** (skipping this is the No. 1 cause of `python` being "not recognized" in the terminal).
2. *(Optional — for the R package)* install R from **[cran.r-project.org/bin/windows/base](https://cran.r-project.org/bin/windows/base/)**.
3. Open **PowerShell** and install morie:

```powershell
python -m pip install --upgrade pip
python -m pip install morie
python -c "import morie; print(morie.__version__)"
morie interactive install
```

For the R package, install **rmorie** (the R distribution of morie): `Rscript -e "install.packages('rmorie', repos=c('https://rootcoder007.r-universe.dev','https://cloud.r-project.org'))"`

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
morie interactive install
```

The tap repo is [`rootcoder007/homebrew-morie`](https://github.com/rootcoder007/homebrew-morie). It pulls morie's source distribution from PyPI and bundles a self-contained `python@3.12` venv — no system Python required.

### Python — PyPI (manual; requires `pip` already installed)

```bash
pip install morie
morie interactive install
```

> **Heads-up:** Debian, Ubuntu and Raspberry Pi OS forbid `pip` outside a virtual environment (PEP 668). Use the venv three-liner above, or the one-liner installer, which sets one up for you. morie needs no NumPy or SciPy, so no compiled scientific stack has to be present.

### Python — Docker (no local dependencies)

```bash
# Latest stable
docker run --rm ghcr.io/rootcoder007/morie:latest morie --help

# Pin to a specific version (recommended for reproducibility)
docker run --rm ghcr.io/rootcoder007/morie:1.4.0 morie --help
```

Published on every release with a versioned tag, a major.minor tag and `:latest` (linux/amd64). Requires only Docker — no Python, no pip.

### The interactive layer — `morie repl`, `morie exec`, `morie agent`, `morie tui`

Every channel above leaves five modules out on purpose: they execute code that
you or a model type, and package scanners flag that surface. Add them for your
user in one step (the one-liner installer does it for you; a verb that needs
them prints this command and, on a terminal, offers to run it):

```bash
morie interactive install
```

That fetches the files for your installed version from the tagged source on
GitHub, checks each one against the SHA-256 manifest shipped inside the
package, and stores them under `~/.local/share/morie/interactive`
(`%LOCALAPPDATA%\morie\interactive` on Windows). Nothing in site-packages
changes. `morie interactive status` shows what is active and `morie interactive
remove` takes it out again. The TUI also needs `pip install "morie[interactive]"`.
A source checkout has all of this already. Offline: `morie interactive install
--from path/to/morie/src/morie`.

### R package: rmorie — r-universe

The R distribution of morie is the **[rmorie](https://github.com/rootcoder007/rmorie)** package.

```r
# rmorie comes from r-universe (prebuilt binaries for macOS and Windows,
# source on Linux); its companions rmoriebricklayer and rmoriedata are on CRAN
install.packages(
  "rmorie",
  repos = c(
    rootcoder007 = "https://rootcoder007.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
install.packages(c("rmoriebricklayer", "rmoriedata"))

# The same R code also lives in this repository as r-package/morie (package
# name "morie"), tracking every commit here; build it from source with remotes
# (needs a C++ toolchain and rmoriebricklayer):
# install.packages("remotes")
remotes::install_github("rootcoder007/morie", subdir = "r-package/morie")
```

Or let morie run either install for you: `morie r-install` (r-universe) or
`morie r-install --github` (this repository's R arm).

## Quick start

```python
import morie
from morie.data import load_dataset

# CPADS 2021-22 public-use microdata (40,931 rows): downloaded from
# open.canada.ca on first use (about a minute), then served from the
# local cache in ~/.cache/morie
df = load_dataset("ocp21")
print(df.shape)

# Welch's two-sample t-test, and the guide every morie.fn callable carries
from morie.fn import describe, welcht

result = welcht([5.1, 4.9, 5.6, 5.8, 6.0], [6.2, 6.8, 7.1, 6.5, 7.4])
print(result)
print(describe("welcht"))
```

### From the terminal

Every feature is a verb of the `morie` command; `morie --help` lists them,
`morie cheatsheet` fits them on one page, and the R package ships the same
verbs as `rmorie` (see below).

```bash
morie list-modules                                   # the 23 analysis modules
morie run-module power-design --output-dir out/      # one module, its tables as CSV
morie explain power_two_proportion_gender.csv        # how to read an output table
morie list-datasets                                  # 71 catalog keys + the curated tables after login
morie pull ocp21 --out cpads.csv                     # the real CPADS PUMF, cached; modules use it from then on
morie pull --all --out datasets/                     # every catalog dataset
morie login                                          # one key for the hosted model tier + data.rmorie.com, with a GitHub account
morie login --email you@example.com                  # no GitHub account: a code is emailed, type it at the prompt
morie login --no-browser                             # server / SSH / no browser: prints a link + code for any device
morie pull chicago_crime/incidents --out incidents.csv   # a curated table (8.6M rows)
morie ask "which module fits a treatment-control design?"
morie provider set --base-url https://api.openai.com/v1 --key sk-...   # or your own model endpoint
morie emissions --seconds 5 --country CAN            # energy + CO2 of this machine, sealed in a capsule
morie pipeline --modules power-design -y             # modules + emissions tracking + capsule
morie verify-pollution --pollutant no2 --demo        # pollution -> health causal pipeline
morie selftest                                       # every subsystem, offline (no downloads)
```

### Where the datasets come from

`morie list-datasets` prints a **Route** column next to every key. Of the
71 keys, 70 download themselves on first use and are cached: open.canada.ca
(CPADS, CSADS, CSUS microdata and bootstrap weights), data.ontario.ca (the
OTIS correctional tables and institution locations), Statistics Canada
(CCHS), CIHI (the indicator library and its tables), Environment Canada's
NAPS air-quality files, the Toronto Police ArcGIS feeds (the Canada-wide
NAPS hourly keys are about 2 million rows: allow ten minutes and 1.5 GB),
the Health Infobase tables from health-infobase.canada.ca with the
data.rmorie.com copy as the fallback, the OTIS research environments from
data.rmorie.com (R objects: rmorie loads them, morie saves them for R), and
the reviewed SIU corpus from the
[rmoriedata](https://cran.r-project.org/package=rmoriedata) package on CRAN
(fetched as a source tarball, no R needed). One key, the MAPQ workbook, is
your own file: put it under a data directory and point `MORIE_DATA_DIR` at
it, keeping the relative path that `morie list-datasets` shows. The curated
tables at data.rmorie.com join the list after `morie login` (GitHub) or
`morie login --email you@example.com`.

```python
from morie.data import list_rmoriedata, load_rmoriedata
[r["slug"] for r in list_rmoriedata()]        # 99 tables + 8 data dictionaries
siu = load_rmoriedata("siu_directors_reports")  # 5,157 reviewed SIU reports
```

Beyond the catalog, the project keeps **160 databases materialised from
Google BigQuery public datasets, plus the Health Infobase tables and the OTIS research files** (Chicago crime, EPA air quality, US
census, FEC, FDA, NOAA, NHTSA, Hacker News, Ethereum, World Bank, ...),
served from the edge at <https://data.rmorie.com> and opened by the key
`morie login` stores. `morie list-datasets` shows their `db/table` keys
with the route "data.rmorie.com"; `morie pull db/table` and
`load_dataset("db/table")` fetch one (cached locally), and
`https://data.rmorie.com/browse` runs SQL on any of them in the browser.
They are rebuilt weekly without any project machine in the loop.

### The R side has every verb

`rmorie` (r-universe) ships the same command line: `rmorie::install_cli()`
puts `rmorie` on your PATH, and `rmorie pull ocp21`, `rmorie run-module`,
`rmorie emissions`, `rmorie verify-pollution`, `rmorie provider set`,
`rmorie selftest` and the rest behave like their `morie` twins. The two
packages share the credentials file, the CPADS resolution order, the
emissions CSV layout and the capsule format, so they mix in one workflow.
`morie r-install` installs the R side from Python.

## What's new

Per-release user-facing changes are now in [WHATS_NEW.md](https://github.com/rootcoder007/morie/blob/main/WHATS_NEW.md).
For the R-package changelog see [r-package/morie/NEWS.md](https://github.com/rootcoder007/morie/blob/main/r-package/morie/NEWS.md).
For the planned roadmap see [ROADMAP.md](https://github.com/rootcoder007/morie/blob/main/ROADMAP.md).

## Documentation

Full documentation is at [rootcoder007.github.io/morie](https://rootcoder007.github.io/morie/).

- **Website**: <https://rmorie.com> — the MORIE family (rmorie, morie, rmoriebricklayer, rmoriedata) in one place.
- **Curated data**: <https://data.rmorie.com> — the BigQuery-built tables, opened by the same key; `/browse` for SQL in the browser.
- **CLI reference**: [cli](https://rootcoder007.github.io/morie/cli.html); learn pages for [datasets](https://rootcoder007.github.io/morie/learn/datasets.html), [emissions and capsules](https://rootcoder007.github.io/morie/learn/emissions.html) and [the R command line](https://rootcoder007.github.io/morie/learn/r-cli.html).
- **Hosted LLM tier**: <https://llm.rmorie.com> — the fallback model behind `morie ask` when there is no local Ollama. Sign in with `morie login` (GitHub) or `morie login --email you@example.com`; on a server or over SSH, `morie login --no-browser` prints a link and a code to open on any device (1.4.0 opens no browser there on its own); `morie models` lists the models on your key and `morie ask --model NAME` picks one; see the [hosted tier docs](https://rootcoder007.github.io/morie/hosted.html). The tier serves ollama.com cloud models and additional AI models (kimi-k2.6:cf, kimi-k2.7-code:cf, deepseek-v4-pro:cf, deepseek-v4-flash:cf, glm-5.2:cf, glm-5.3:cf, glm-5.3-flash:cf, gpt-oss-120b:cf, gpt-oss-20b:cf, llama-4-scout:cf, qwen3.8-27b:cf, nemotron-3-120b:cf and gemma-4-26b:cf); a rate-limited cloud model falls back to one of them.

## Citation

If you use morie in your research, please cite the software:

> Ruhela, V. S. (2026). *morie: Multi-domain Open Research and Inferential Estimation.* https://github.com/rootcoder007/morie

BibTeX:

```bibtex
@Manual{ruhela_morie_2026,
  title  = {morie: Multi-domain Open Research and Inferential Estimation},
  author = {Vansh Singh Ruhela},
  year   = {2026},
  url    = {https://github.com/rootcoder007/morie},
}
```

The single citation above covers both the R (`r-package/morie/`) and Python (`src/morie/`) implementations, which ship under the same version.

See [`CITATION.cff`](https://github.com/rootcoder007/morie/blob/main/CITATION.cff)
for machine-readable citation metadata (BibTeX, etc.) — that file is
what GitHub's "Cite this repository" button consumes.


## Acknowledgments

### AI assistance

MORIE was developed with substantial assistance from frontier AI
assistants. The author retains full responsibility for the code, the
methods, and the scientific claims; AI assistance accelerated
implementation but does not change the attribution of the work.

- **Claude — Anthropic.** Anthropic's Claude family (Opus, Sonnet, and
  Haiku across the 4.x and 5.x generations) was used extensively throughout
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

### Data acknowledgments

Several MRM analyses use Statistics Canada and Health Canada Public
Use Microdata Files (PUMFs) — including the **Canadian Cannabis
Survey (CCS)**, the **Canadian Student Alcohol and Drugs Survey
(CSADS)**, the **Canadian Substance Use Survey (CSUS)**, the
**Canadian Alcohol and Drugs Survey (CADS)**,
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

## Trust model — what the powerful features can do to your machine

morie is a developer/research toolkit with some intentionally powerful
surfaces. They run **local, user-supplied** input by design — the only network-supplied
code is the verified interactive layer below — but under an untrusted-input threat model each is
high-impact, so every one defaults to the safe behaviour and gates the risky
path behind a single, named, off-by-default environment knob:

| Surface | What it can do | Safe default → opt-in |
|---|---|---|
| `MORIE_NO_EXEC` | master kill-switch for all dynamic execution | executes; set `MORIE_NO_EXEC=1` to disable REPL/exec/shell everywhere |
| **`morie tui`** / **polyglot** REPLs | run Python/R/shell/other code you type, like any REPL | run your input only; honour `MORIE_NO_EXEC=1`. Feed them only code you'd run at your own shell |
| **`morie interactive install`** | fetches the REPL/exec/agent/TUI modules (five files) from this version's release tag on GitHub, the one network-supplied code path | every file is checked against the SHA-256 manifest shipped inside the package before it is enabled; nothing runs during the install; `--no-verify` is the opt-in for a non-release `--ref` |
| **`pt2gguf`** | deserialize `.pt`/`.pkl` model files | `torch.load(weights_only=True)` + a restricted unpickler; the code-executing `weights_only=False` path needs `MORIE_TRUST_CHECKPOINT=1`. Use trusted checkpoints only |
| **`bin/morie`** installer — Ollama | remote `install.sh` | fetched to a temp file with its SHA256 printed; executes only under `MORIE_ALLOW_REMOTE_INSTALL=1` |
| **`bin/morie`** — `ESML_RC` config | shell config sourced on every run (persistence) | **not sourced** unless `MORIE_ALLOW_RC=1` |
| **`bin/morie`** — `morie cron` | writes your crontab (persistence) | `add`/`remove` refuse unless `MORIE_ALLOW_CRON=1`; `list` is read-only |
| **`bin/morie`** — `backup restore` | extracts a tar archive | archives with absolute or `..` paths are refused before extraction |

`morie doctor` prints the current state of every knob so you can see the active
trust posture at a glance. Only enable a knob for inputs you fully control.

## Reporting issues / security

- General issues: [GitHub Issues](https://github.com/rootcoder007/morie/issues)
- Security vulnerabilities: see [`SECURITY.md`](https://github.com/rootcoder007/morie/blob/main/.github/SECURITY.md)
