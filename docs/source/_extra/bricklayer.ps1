# SPDX-License-Identifier: AGPL-3.0-or-later
#Requires -Version 5.0
<#
.SYNOPSIS
    bricklayer.ps1 -- assemble the morie family on Windows.

.DESCRIPTION
    The Windows counterpart of bricklayer.sh. Works "for everyone" --
    even if the user has no pip (or no Python at all): it offers to
    install Python via winget, then installs the morie family.

    Open packages it offers to install:
      * morie   -- Python package (pip)
      * rmorie  -- R package (r-universe; pulls rmoriedata + rmoriebricklayer)
    The proprietary rmorie-cli is never auto-installed -- only pointed to.

    The family is built on a shared C/C++ numeric core (libmorie ->
    morie._core in Python; rmoriebricklayer's compiled kernels in R). This
    script detects a C/C++ toolchain and verifies the compiled backend
    actually loaded, warning loudly that the project is degraded/slow
    without it. On Windows the R side needs Rtools to build from source.

.PARAMETER Yes
    Install without prompting (assume yes). Same as $env:MORIE_BRICKLAYER_YES=1.

.PARAMETER Check
    Report status only; install nothing.

.EXAMPLE
    irm https://rootcoder007.github.io/morie/bricklayer.ps1 | iex
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File bricklayer.ps1 -Yes
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$Check
)

$ErrorActionPreference = 'Continue'

if ($env:MORIE_NO_BRICKLAYER -eq '1') { exit 0 }
if ($env:MORIE_BRICKLAYER_YES -eq '1') { $Yes = $true }

$RUNIV   = 'https://rootcoder007.r-universe.dev'
$CRAN    = 'https://cloud.r-project.org'
$CLI_URL = 'https://github.com/rootcoder007/rmorie-cli'

$script:PyExe = $null
$script:PyPrefix = @()

function Have-Cmd($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Resolve-Python {
    if (Have-Cmd 'py')      { $script:PyExe = 'py';      $script:PyPrefix = @('-3'); return $true }
    if (Have-Cmd 'python')  { $script:PyExe = 'python';  $script:PyPrefix = @();     return $true }
    if (Have-Cmd 'python3') { $script:PyExe = 'python3'; $script:PyPrefix = @();     return $true }
    return $false
}

function Invoke-Py([string[]]$PyArgs) {
    & $script:PyExe @($script:PyPrefix + $PyArgs) 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Have-PyMorie {
    if (-not $script:PyExe) { return $false }
    return (Invoke-Py @('-c', 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("morie") else 1)'))
}
function Py-Backend-Ok {
    if (-not $script:PyExe) { return $false }
    return (Invoke-Py @('-c', 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("morie._core") else 1)'))
}
function Have-Pip {
    if (-not $script:PyExe) { return $false }
    return (Invoke-Py @('-m', 'pip', '--version'))
}
function Have-RMorie {
    if (-not (Have-Cmd 'Rscript')) { return $false }
    & Rscript -e 'quit(status=as.integer(!requireNamespace("rmorie",quietly=TRUE)))' 2>$null
    return ($LASTEXITCODE -eq 0)
}
function R-Backend-Ok {
    if (-not (Have-Cmd 'Rscript')) { return $false }
    & Rscript -e 'quit(status=as.integer(!isTRUE(rmorie::morie_fast_available())))' 2>$null
    return ($LASTEXITCODE -eq 0)
}
function Have-Toolchain {
    return ((Have-Cmd 'cl') -or (Have-Cmd 'gcc') -or (Have-Cmd 'clang'))
}

function Confirm-Yes($msg) {
    if ($Yes) { return $true }
    if (-not [Environment]::UserInteractive) { return $false }
    $r = Read-Host "$msg [Y/n]"
    return ($r -eq '' -or $r -match '^(y|yes)$')
}

function Mark($ok, $label) {
    if ($ok) { Write-Host "  [x] $label" } else { Write-Host "  [ ] $label" }
}

# ---- detect + report ------------------------------------------------------

$hasPython = Resolve-Python
$pyOk  = if ($hasPython) { Have-PyMorie } else { $false }
$rOk   = Have-RMorie
$cliOk = Have-Cmd 'rmorie'
$tcOk  = Have-Toolchain

Write-Host "morie family status:"
Mark $pyOk  "morie            (Python / pip)"
Mark $rOk   "rmorie + data + bricklayer  (R / r-universe)"
Mark $cliOk "rmorie-cli       (proprietary -- not auto-installed)"
Mark $tcOk  "C/C++ toolchain  (Rtools / MSVC -- needed to build from source)"

if ($pyOk -and -not (Py-Backend-Ok)) {
    Write-Host "  !! morie is installed but its C++ backend (morie._core) is NOT active -- degraded."
}
if ($rOk -and -not (R-Backend-Ok)) {
    Write-Host "  !! rmorie is installed but its C/C++ kernels are NOT active (slow pure-R fallback)."
    Write-Host "     On Windows, R builds packages from source with Rtools: https://cran.r-project.org/bin/windows/Rtools/"
}
Write-Host ""

if ($Check) { exit 0 }

if (-not $cliOk) {
    Write-Host "note: rmorie-cli is proprietary (Receipt-of-Custody); obtain it at $CLI_URL"
}

# ---- Python: install Python itself if missing, then morie -----------------

if (-not $pyOk) {
    if (-not $hasPython) {
        Write-Host "Python is not installed."
        if ((Have-Cmd 'winget') -and (Confirm-Yes "Install Python now via winget?")) {
            winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
            # winget updates PATH for new shells; re-resolve for this session best-effort.
            $hasPython = Resolve-Python
        }
        if (-not (Resolve-Python)) {
            Write-Host "Install Python from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run this script."
        }
    }
    if ($script:PyExe) {
        if (-not (Have-Pip)) {
            Write-Host "-> bootstrapping pip (ensurepip) ..."
            Invoke-Py @('-m', 'ensurepip', '--upgrade') | Out-Null
        }
        if (Have-Pip -and (Confirm-Yes "Install the Python package morie now?")) {
            Write-Host "-> installing Python morie ..."
            & $script:PyExe @($script:PyPrefix + @('-m', 'pip', 'install', '--upgrade', 'morie'))
            if ($LASTEXITCODE -ne 0) { Write-Host "Python morie install failed." -ForegroundColor Red }
            elseif (-not (Py-Backend-Ok)) {
                Write-Host "WARNING: morie installed but morie._core (C++ backend) did not load -- it will be degraded." -ForegroundColor Yellow
            }
        }
    }
}

# ---- R: install rmorie (cascades rmoriedata + rmoriebricklayer) ------------

if (-not $rOk) {
    if (Have-Cmd 'Rscript') {
        if (Confirm-Yes "Install the R package rmorie (+ rmoriedata, rmoriebricklayer) now?") {
            Write-Host "-> installing R rmorie ..."
            & Rscript -e "install.packages('rmorie', repos=c('$RUNIV','$CRAN'))"
            if ($LASTEXITCODE -ne 0) { Write-Host "R rmorie install failed." -ForegroundColor Red }
            elseif (-not (R-Backend-Ok)) {
                Write-Host "WARNING: rmorie installed but its C/C++ kernels are inactive -- install Rtools and reinstall." -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "R not found. Install R from $CRAN, then re-run, or:"
        Write-Host "  Rscript -e `"install.packages('rmorie', repos=c('$RUNIV','$CRAN'))`""
    }
}

Write-Host ""
Write-Host "Done. (Re-run with -Check to see status.)"
