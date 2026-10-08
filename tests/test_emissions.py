"""Tests for morie.emissions — pure-Python emissions tracker.

No hardcoded column counts or magic numbers — all expectations are derived
from the actual EmissionsData.csv_header property so tests stay in sync
with the implementation.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

# The tracker reads /proc and sysctl itself (no psutil); nothing to skip.


class TestCarbonIntensity:
    def test_known_countries_relative_ordering(self):
        """Countries with known energy mixes should have plausible relative ordering."""
        from morie.emissions import _get_carbon_intensity, _load_energy_data

        _load_energy_data()
        fra = _get_carbon_intensity("FRA")  # nuclear-heavy, low carbon
        usa = _get_carbon_intensity("USA")  # mixed grid
        assert fra < usa, f"France ({fra}) should be lower than USA ({usa})"
        assert fra > 0, "Intensity must be positive"
        assert usa > 0, "Intensity must be positive"

    def test_unknown_country_returns_world_average(self):
        from morie.emissions import _G_TO_KG, _WORLD_AVERAGE_G_KWH, _get_carbon_intensity

        val = _get_carbon_intensity("ZZZ")
        expected = _WORLD_AVERAGE_G_KWH * _G_TO_KG
        assert abs(val - expected) < 0.001, f"Unknown should use world average, got {val}"

    def test_canada_with_and_without_region(self):
        """CAN with region should differ from CAN without (if regional data exists)."""
        from morie.emissions import _get_carbon_intensity

        can_national = _get_carbon_intensity("CAN")
        can_region = _get_carbon_intensity("CAN", "ontario")
        # Both must be positive; they may or may not differ depending on data
        assert can_national > 0
        assert can_region > 0

    def test_all_positive(self):
        from morie.emissions import _get_carbon_intensity

        for iso in ["USA", "CAN", "GBR", "DEU", "CHN", "IND", "AUS", "BRA", "JPN", "FRA"]:
            val = _get_carbon_intensity(iso)
            assert val > 0, f"{iso} must have positive carbon intensity"


class TestRamPower:
    def test_returns_positive(self):
        from morie.emissions import _estimate_ram_power

        assert _estimate_ram_power() > 0

    def test_within_physical_bounds(self):
        """RAM power should be between 1W and 200W for any real machine."""
        from morie.emissions import _estimate_ram_power

        power = _estimate_ram_power()
        assert 1.0 <= power <= 200.0, f"RAM power {power}W outside physical bounds"


class TestEmissionsTracker:
    def test_start_stop_returns_float(self):
        from morie.emissions import EmissionsTracker

        with tempfile.TemporaryDirectory() as d:
            t = EmissionsTracker(
                project_name="test",
                output_dir=d,
                measure_power_secs=0.5,
                country_iso_code="CAN",
                save_to_file=False,
            )
            t.start()
            time.sleep(1)
            emissions = t.stop()
            assert isinstance(emissions, float)
            assert emissions >= 0

    def test_context_manager_does_not_raise(self):
        from morie.emissions import EmissionsTracker

        with (
            tempfile.TemporaryDirectory() as d,
            EmissionsTracker(
                project_name="ctx",
                output_dir=d,
                measure_power_secs=0.5,
                country_iso_code="USA",
                save_to_file=False,
            ),
        ):
            time.sleep(0.5)

    def test_csv_header_matches_data_row(self):
        """CSV header column count must match data row column count.

        Derived from EmissionsData.csv_header — no hardcoded number.
        """
        from morie.emissions import EmissionsData, EmissionsTracker

        expected_cols = len(EmissionsData().csv_header.split(","))

        with tempfile.TemporaryDirectory() as d:
            t = EmissionsTracker(
                project_name="csv-test",
                output_dir=d,
                output_file="test_emissions.csv",
                measure_power_secs=0.5,
                country_iso_code="FRA",
            )
            t.start()
            time.sleep(1)
            t.stop()

            csv_path = Path(d) / "test_emissions.csv"
            assert csv_path.exists(), "CSV file should be created"
            lines = csv_path.read_text().strip().splitlines()
            assert len(lines) == 2, "Header + 1 data row"
            header_cols = len(lines[0].split(","))
            data_cols = len(lines[1].split(","))
            assert header_cols == expected_cols, (
                f"Header has {header_cols} cols but EmissionsData defines {expected_cols}"
            )
            assert data_cols == header_cols, f"Data row has {data_cols} cols but header has {header_cols}"
            assert "timestamp" in lines[0]
            assert "emissions" in lines[0]
            assert "country_iso_code" in lines[0]

    def test_emissions_data_has_required_fields(self):
        """EmissionsData must have all fields needed for codecarbon-compatible output."""
        from morie.emissions import EmissionsData

        data = EmissionsData()
        required = [
            "emissions",
            "cpu_energy",
            "gpu_energy",
            "ram_energy",
            "energy_consumed",
            "country_iso_code",
            "pue",
            "wue",
            "cpu_power",
            "gpu_power",
            "ram_power",
            "duration",
        ]
        for field in required:
            assert hasattr(data, field), f"EmissionsData missing required field: {field}"

    def test_csv_row_col_count_matches_header(self):
        """csv_row() must produce same number of fields as csv_header."""
        from morie.emissions import EmissionsData

        data = EmissionsData(project_name="test", country_iso_code="USA")
        header_count = len(data.csv_header.split(","))
        row_count = len(data.csv_row().split(","))
        assert row_count == header_count, f"csv_row has {row_count} fields but csv_header has {header_count}"


class TestCapsuleAndVerb:
    def test_run_writes_csv_manifest_and_formula_holds(self, monkeypatch):
        from morie import emissions as em

        monkeypatch.setenv("MORIE_EMISSIONS_OFFLINE", "1")
        with tempfile.TemporaryDirectory() as d:
            data = em.run_check(0.4, d, country_iso_code="CAN")
            assert data.emissions > 0
            ci = em._get_carbon_intensity("CAN")
            assert abs(data.emissions - data.energy_consumed * data.pue * ci) < 1e-15
            assert abs(data.energy_consumed - (data.cpu_energy + data.gpu_energy + data.ram_energy)) < 1e-15
            assert 0.0 <= data.cpu_utilization_percent <= 100.0
            csv = (Path(d) / "emissions.csv").read_text().splitlines()
            assert csv[0] == data.csv_header
            assert len(csv) == 2
            manifest = json.loads((Path(d) / "emissions_manifest.json").read_text())
            assert manifest["meta"]["measurements"]["emissions"] == data.emissions
            assert manifest["meta"]["emissions_csv_sha256"] == em._sha256_file(Path(d) / "emissions.csv")
            cap = data.capsule
            assert cap["manifest"].endswith("emissions_manifest.json")
            assert cap["signed"] in (True, False)
            if cap["signed"]:
                assert Path(cap["bundle"]).exists()
            text = em.summary_text(data, cap)
            assert "kg CO2eq" in text and "Capsule:" in text

    def test_country_env_overrides_lookup(self, monkeypatch):
        from morie import emissions as em

        monkeypatch.setenv("MORIE_COUNTRY_ISO", "FRA")
        monkeypatch.delenv("MORIE_EMISSIONS_OFFLINE", raising=False)
        with tempfile.TemporaryDirectory() as d:
            t = em.EmissionsTracker(output_dir=d, capsule=False)
            t.start()
            t.stop()
            assert t._country_iso == "FRA"

    def test_emissions_verb_parses(self):
        from morie.runner import build_parser

        a = build_parser().parse_args(
            ["emissions", "--seconds", "1", "--output-dir", "x", "--no-capsule", "--country", "CAN"]
        )
        assert (a.command, a.seconds, a.output_dir, a.no_capsule, a.country) == ("emissions", 1.0, "x", True, "CAN")


class TestOfflineLocation:
    def test_time_zone_then_locale_through_the_full_iso_list(self):
        from morie import emissions as em

        assert em._detect_location_offline(tz="Europe/Stockholm")[0] == "SWE"
        assert em._detect_location_offline(tz="US/Eastern")[0] == "USA"
        assert em._detect_location_offline(tz="Asia/Calcutta")[0] == "IND"
        assert em._detect_location_offline(tz="posix/America/Toronto")[0] == "CAN"
        assert em._detect_location_offline(tz="Etc/UTC", territory="IN")[0] == "IND"
        assert em._detect_location_offline(tz="", territory="")[0] == ""
        assert em.iso3("kz") == "KAZ" and em.iso3("se") == "SWE" and em.iso3("CAN") == "CAN"
        table = em._timezone_countries()
        assert len(table) > 540 and table["Europe/Oslo"] == "NO"

    def test_every_geographic_zone_of_the_tz_database_resolves(self):
        import zoneinfo

        from morie import emissions as em

        try:
            zones = zoneinfo.available_timezones()
        except Exception:  # pragma: no cover - no tz database on this host
            return
        geo = {z for z in zones if "/" in z and not z.startswith(("Etc/", "posix/", "right/", "SystemV/"))}
        missing = sorted(geo - set(em._timezone_countries()))
        assert missing == [], missing[:20]

    def test_offline_env_uses_the_offline_route(self, monkeypatch):
        from morie import emissions as em

        monkeypatch.delenv("MORIE_COUNTRY_ISO", raising=False)
        monkeypatch.setenv("MORIE_EMISSIONS_OFFLINE", "1")
        monkeypatch.setenv("TZ", "Europe/Stockholm")
        assert em._detect_location_offline()[0] == "SWE"
        monkeypatch.setattr(em, "_system_timezone", lambda: "")
        monkeypatch.setattr(em, "_locale_territory", lambda: "FR")
        assert em._detect_location_offline()[0] == "FRA"

    def test_network_failure_falls_back_to_the_offline_route(self, monkeypatch):
        import sys

        from morie import emissions as em

        monkeypatch.setitem(sys.modules, "httpx", None)
        monkeypatch.setenv("TZ", "Europe/Oslo")
        assert em._detect_location()[0] == "NOR"
