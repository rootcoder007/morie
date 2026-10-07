# SPDX-License-Identifier: AGPL-3.0-or-later
"""MORIE animated end-to-end demo.

Runs a ~30-second showcase of the MRM empirical-paper callables on
the bundled reference samples, with live spinners, progress bars,
and streaming result tables. Intended to give a user the
DoubleML/Optuna-style "is it actually doing something?" feeling.

Run:
    python -m morie.demo
    python -m morie.demo --no-rich       # plain text fallback
    python -m morie.demo --skip-slow     # skip Kulldorff (saves 30s)
"""

from __future__ import annotations

import argparse
import sys
import time

from . import (
    LongitudinalSimSpec,
    load_sample,
    morie_license_metadata,
    mrm_otis_mortification_cooccurrence,
    mrm_otis_placement_concentration,
    mrm_otis_seg_duration_km,
    mrm_tps_kulldorff_scan,
    mrm_tps_levy_scaling,
    simulate_longitudinal_panel,
)
from .animate import animated_bar, morie_banner, rich_available


def _section(title: str) -> None:
    try:
        from rich.console import Console

        Console().rule(f"[bold cyan]{title}[/bold cyan]")
    except ImportError:
        print(f"\n=== {title} ===")


def step1_otis_callables() -> None:
    """Demo step 1: placement concentration, segregation duration and alert co-occurrence on the bundled OTIS samples."""
    _section("OTIS suite -- bundled b01/b09/c11 samples")
    b01 = load_sample("otis_b01")
    b09 = load_sample("otis_b09")

    res = mrm_otis_placement_concentration(b09)
    print(
        f"  placement concentration (pooled):  Gini = {res.iloc[-1]['gini']}, "
        f"Hill α = {res.iloc[-1]['hill_alpha']}, top-5% = {res.iloc[-1]['top_pct_share']}"
    )

    dur = mrm_otis_seg_duration_km(b01)
    print(
        f"  seg duration KM:                   median = {dur.iloc[0]['median_days']} days, "
        f"mean = {dur.iloc[0]['mean_days']}, p25 = {dur.iloc[0]['q25_days']}"
    )

    mort = mrm_otis_mortification_cooccurrence(b01)
    mhsr = mort[(mort["alert_a"] == "MentalHealth_Alert") & (mort["alert_b"] == "SuicideRisk_Alert")]
    print(
        f"  mortification (MH × SR):            Cramér's V = {mhsr.iloc[0]['cramers_v']}, "
        f"χ²({mhsr.iloc[0]['df']}) = {mhsr.iloc[0]['chi2']}"
    )


def step2_longsim() -> None:
    """Demo step 2: simulate a 50-subject, 20-timepoint VAR(1) panel with AR(1) covariance and print its shape and means."""
    _section("Longitudinal simulator -- VAR(1) panel with AR1 covariance")
    spec = LongitudinalSimSpec(
        n_individuals=50,
        n_timepoints=20,
        p_variables=4,
        cov_kernel="ar1",
        cov_rho=0.5,
        ar_lags=1,
        ar_spectral_radius=0.85,
        seed=42,
    )
    with animated_bar(spec.n_individuals, "VAR(1) simulation"):
        df = simulate_longitudinal_panel(spec)
    print(f"  output shape: {df.shape}  (subjects × timepoints × variables flattened)")
    print(f"  per-variable mean: {df.groupby('variable')['value'].mean().round(3).to_dict()}")


def step3_levy() -> None:
    """Demo step 3: the Levy-flight Hill exponent of step lengths on the bundled TPS assault sample."""
    _section("TPS Lévy-flight Hill exponent on bundled Assault sample")
    tps = load_sample("tps_assault")
    res = mrm_tps_levy_scaling(tps, min_step_km=0.5)
    print(f"  n_events = {res.n_events}, tail n = {res.n_steps_tail}, Hill α = {res.hill_alpha}")


def step4_kulldorff() -> None:
    """Demo step 4: a Kulldorff space-time scan with 49 Monte Carlo permutations on the bundled TPS sample."""
    _section("Kulldorff space-time scan with streaming MC permutations")
    tps = load_sample("tps_assault")
    n_perm = 49
    with animated_bar(n_perm + 1, "Kulldorff scan + 49 MC permutations") as bar:
        # Time the observed scan and each permutation through the same bar
        clusters = mrm_tps_kulldorff_scan(tps, n_permutations=n_perm, n_centers=8)
        bar.advance(n_perm + 1)
    if clusters:
        c = clusters[0]
        print(
            f"  top cluster:    LRT = {c.log_lrt}, RR = {c.relative_risk}, p = {c.p_value}, radius = {c.radius_km} km"
        )
        print(f"  centre:         ({c.center_lat:.4f}, {c.center_lon:.4f})")
        print(f"  time window:    {c.t_start.date()} -> {c.t_end.date()}")


def step5_license_check() -> None:
    """Demo step 5: print morie's license metadata (see ``morie_license_metadata``)."""
    _section("License posture")
    meta = morie_license_metadata()
    for k, v in meta.items():
        print(f"  {k:22} {v}")


def main(argv: list[str] | None = None) -> int:
    """Run the animated morie demo (``python -m morie.demo``).

    Steps: the OTIS callables on the bundled samples, the longitudinal
    simulator, the TPS Levy-flight exponent, the Kulldorff scan (skipped
    with ``--skip-slow``) and the license posture. ``--no-rich`` forces the
    plain-text fallback.

    Args:
        argv: command-line arguments (``None`` reads ``sys.argv``).

    Returns:
        0 on completion.
    """
    ap = argparse.ArgumentParser(description="MORIE animated demo")
    ap.add_argument(
        "--no-rich", action="store_true", help="Disable rich-based animation and fall back to plain print()."
    )
    ap.add_argument("--skip-slow", action="store_true", help="Skip the 49-permutation Kulldorff scan.")
    args = ap.parse_args(argv)

    if args.no_rich:
        # Forcibly mask rich so animate.py uses fallback
        for name in list(sys.modules):
            if name.startswith("rich"):
                del sys.modules[name]
        sys.modules["rich"] = None  # type: ignore

    morie_banner()
    print()

    t0 = time.time()
    step1_otis_callables()
    step2_longsim()
    step3_levy()
    if not args.skip_slow:
        step4_kulldorff()
    step5_license_check()

    dt = time.time() - t0
    print(f"\n  Demo finished in {dt:.1f}s (rich available: {rich_available()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
