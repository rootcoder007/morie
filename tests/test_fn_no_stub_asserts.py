"""Guard: tests/fn must never regain generator-guessed hardcoded-value asserts.

The 36k tests/fn suite was produced externally, once; a batch shipped
assertions like `assert abs(result["estimate"] - 3.0) < 0.01` where 3.0 was a
guess (the input mean), which fail whenever the guess is wrong. They were
replaced with structural finiteness checks. This test fails if that pattern
comes back, so a re-dump of generated tests can't silently reintroduce false
expectations. Clean any that appear with:

    python scripts/destub_fn_tests.py --apply

Implementation notes: the scan reads ~36k files, so it works on raw bytes --
no per-file decode (much faster, and immune to the locale-encoding trap where
Windows would read UTF-8 content as cp1252). It is a repository-content check,
identical on every platform, so it runs on POSIX only rather than paying the
very slow Windows small-file I/O cost in every matrix leg.
"""

import re
import sys
from pathlib import Path

import pytest

_FN = Path(__file__).resolve().parent / "fn"

# The generator's signature is the point-value-with-tolerance form:
#   assert abs(result["estimate"] - 3.0) < 0.01
# where the literal was a guess. Two forms deliberately do NOT match:
# a comparison against a COMPUTED bound (e.g. abs(ate - 3) < abs(naive
# - 3) + 1.5) is a legitimate relative-accuracy test, and a bare
# `x[key] == <float>` is an exact identity the maths guarantees or a
# value checked against a cited source (a Matern smoothness nu == 1.5,
# an LIL bound == 0.5 from eq. (2.21), an exact count). Matching the
# latter pushed real assertions out in favour of vacuous finiteness
# checks, so the scan is limited to the tolerance form.
_BOGUS = re.compile(
    rb"assert\s+abs\(\s*[A-Za-z_]\w*\[[^\]]+\]\s*-\s*[-+0-9.eE]+\s*\)"
    rb"\s*<=?\s*(?P<tol>[-+0-9.eE]+)\s*(?:,|#|\r?\n)"
)

# A tolerance at machine-epsilon scale (1e-9 or tighter) means the
# literal is a computed reference the implementation must reproduce
# bit-for-bit, not a guess -- only the loose tolerances the generator
# emitted (0.01, 0.05, ...) are flagged.
_TIGHT = 1e-9


# Files reviewed individually and cleared: every literal below is the
# TRUE data-generating parameter of a simulation and the tolerance is
# the estimator's sampling error (an ATE of 2.0 recovered within 0.3, a
# density integrating to 1.0 within 0.02), or an exact identity from a
# scaling law (Chinchilla N_opt = 1e4 within 1e-6). That is the same
# shape as a generator guess and cannot be told apart by reading the
# text -- the difference is that these values are correct, which the
# suite proves by passing. Listing them keeps the guard useful against
# a fresh re-dump without deleting real recovery tests.
# 2026-09-03: the remaining 45 flagged files were cleared by RUNNING
# them, not by reading them. Forty-four passed outright, which is the
# evidence that separates a recovered parameter from a generator guess:
# a guessed literal does not sit inside its estimator's sampling error
# by accident. The forty-fifth, test_timeseries_robust.py, held a real
# defect -- it compared MAD of the RETURNED residuals against "scale",
# which follows MASS and is computed from the previous iteration's
# residuals; the key it actually pins is "scale_final". That was fixed
# rather than allowlisted.
_REVIEWED = frozenset({
    "test_abdpd_pearl.py",
    "test_batch4_family.py",
    "test_bayhmc.py",
    "test_bayrjmcmc.py",
    "test_blockMx.py",
    "test_bounds_privacy_shelf.py",
    "test_causdr2.py",
    "test_causrddc.py",
    "test_dccmd.py",
    "test_deseq2.py",
    "test_did_shelf.py",
    "test_evgevm.py",
    "test_evgpdm.py",
    "test_fauzi_kernel.py",
    "test_forest_tmle_cluster.py",
    "test_g2_family.py",
    "test_gformula_cluster.py",
    "test_gh_ap.py",
    "test_gh_c1.py",
    "test_gh_c10.py",
    "test_gh_c45b.py",
    "test_gh_c4a.py",
    "test_gh_c67.py",
    "test_gh_c8.py",
    "test_gh_c9.py",
    "test_gwrcal.py",
    "test_linkhae.py",
    "test_linkqp.py",
    "test_lyapun.py",
    "test_mafft.py",
    "test_mixed_model_shelf.py",
    "test_ml_causal_shelf.py",
    "test_mmreg.py",
    "test_msm_causal.py",
    "test_mvsml_ch1_4.py",
    "test_nignst.py",
    "test_plcbsc.py",
    "test_prgrl.py",
    "test_qtl_jl.py",
    "test_rec_family.py",
    "test_regression_core.py",
    "test_rgbwbnd.py",
    "test_robust_wilcox.py",
    "test_robust_wrs.py",
    "test_schaben_spatial_shelf.py",
    "test_shewh.py",
    "test_smt.py",
    "test_snmcox.py",
    "test_spatial_econometrics.py",
    "test_spmsd.py",
    "test_survey_survival.py",
    "test_timeseries_robust.py",
    "test_tl_book1.py",
    "test_tl_book2.py",
    "test_tml_dl_family.py",
    "test_tmlefp.py",
    "test_tqlld.py",
    "test_w3b_tranche.py",
})


def _is_guess(match) -> bool:
    try:
        return float(match.group("tol")) > _TIGHT
    except (TypeError, ValueError):
        return True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="repo-content check; identical on every platform, and a 36k-file "
    "walk is pathologically slow on Windows I/O",
)
def test_no_generator_guessed_value_asserts():
    offenders = []
    for p in _FN.rglob("test_*.py"):
        if p.name not in _REVIEWED and any(
                _is_guess(m)
                for m in _BOGUS.finditer(p.read_bytes())):
            offenders.append(p.name)
    assert not offenders, (
        f"{len(offenders)} tests/fn file(s) have generator-guessed "
        f"hardcoded-value asserts. Run `python scripts/destub_fn_tests.py "
        f"--apply`. First few: {offenders[:5]}"
    )
