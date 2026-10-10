import sqlite3
from pathlib import Path

import pytest

from morie.fn import _frame_core as pd
from morie.modules import list_modules, run_module


def test_list_modules_includes_core_cpads_steps():
    names = {item["name"] for item in list_modules()}
    assert {
        "power-design",
        "logistic-models",
        "model-comparison",
        "propensity-scores",
        "treatment-effects",
        "ebac-selection-adjustment-ipw",
    }.issubset(names)


def _mock_cpads_df():
    """Minimal CPADS-like DataFrame for testing without the real DB."""
    from morie.fn import _array_core as np

    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "SEQID": range(1, n + 1),
            "weight": rng.uniform(0.5, 2.0, n),
            "alcohol_past12m": rng.choice([0, 1], n),
            "heavy_drinking_30d": rng.choice([0, 1], n),
            "ebac_tot": rng.uniform(0, 0.1, n),
            "ebac_legal": rng.choice([0, 1], n),
            "cannabis_any_use": rng.choice([0, 1], n),
            "age_group": rng.choice(["18-19", "20-22", "23-25"], n),
            "gender": rng.choice(["Female", "Male"], n),
            "province_region": rng.choice(["Ontario", "Quebec", "BC"], n),
            "mental_health": rng.choice(["Good", "Fair", "Poor"], n),
            "physical_health": rng.choice(["Good", "Fair", "Poor"], n),
        }
    )


def _mock_db(tmp_path):
    """Build a tiny SQLite DB with a CPADS-like table."""
    db_path = tmp_path / "mock_morie.db"
    conn = sqlite3.connect(str(db_path))
    # the built-in database keys tables by the catalog table_name ("ocp21");
    # any other name misses and load_dataset falls through to the live portal
    _mock_cpads_df().to_sql("ocp21", conn, index=False)
    conn.close()
    return db_path


def test_load_cpads_from_mock_db(tmp_path, monkeypatch):
    """Load CPADS from a mock SQLite DB (no LFS needed)."""
    from morie.data import load_dataset

    db_path = _mock_db(tmp_path)
    monkeypatch.setattr(
        "morie.data._builtin_db_connect",
        lambda: sqlite3.connect(str(db_path)),
    )
    try:
        frame = load_dataset("ocp21")
    except (ValueError, KeyError):
        pytest.skip("short-key DB resolution unavailable (LFS disabled)")
    assert len(frame) > 0
    assert "SEQID" in frame.columns or "weight" in frame.columns


def test_load_multiple_datasets_mock(tmp_path, monkeypatch):
    """Verify dataset load works with mock DB."""
    from morie.data import load_dataset

    db_path = _mock_db(tmp_path)

    def conn_factory():
        return sqlite3.connect(str(db_path))

    monkeypatch.setattr("morie.data._builtin_db_connect", conn_factory)
    try:
        df = load_dataset("ocp21")
    except (ValueError, KeyError):
        pytest.skip("short-key DB resolution unavailable (LFS disabled)")
    assert len(df) > 0


def test_run_module_with_mock_cpads(tmp_path):
    """Run power-design module using synthetic CPADS CSV."""
    frame = _mock_cpads_df()
    csv_path = tmp_path / "cpads.csv"
    frame.to_csv(csv_path, index=False)
    outputs = run_module("power-design", cpads_csv=str(csv_path), output_dir=tmp_path)
    assert "power_summary" in outputs
    assert (tmp_path / "power_summary.csv").exists()


def test_run_module_materialises_dataset_key_and_cached_pumf_for_the_r_bridge(monkeypatch, tmp_path):
    """--dataset KEY and a pulled PUMF must reach the R stage as a CSV path (they used to be dropped)."""
    from morie import modules
    from morie.fn import _frame_core as pd

    seen = {}

    def fake_bridge(module_name, cpads_csv=None, output_dir=None):
        seen["csv"] = str(cpads_csv)
        seen["frame"] = pd.read_csv(cpads_csv)  # read now: the staged copy is removed when the run ends
        return {"ok": pd.DataFrame({"x": [1]})}

    monkeypatch.setattr(modules, "_run_r_module", fake_bridge)
    monkeypatch.setattr(modules, "_r_route_ready", lambda **_: None)  # the bridge is faked; R need not exist
    monkeypatch.setattr("morie.data.load_dataset", lambda key, **kw: pd.DataFrame({"k": [key]}))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    modules.run_module("descriptive-statistics", dataset_key="ocp21")
    staged = Path(seen["csv"])
    assert staged.name == "ocp21.csv" and staged.parent.name.startswith("morie-dataset-")
    assert seen["frame"]["k"].tolist() == ["ocp21"]
    assert not staged.parent.exists()  # a private directory, removed after the run
    # a real PUMF already in the Python store beats the shipped synthetic frame
    monkeypatch.setattr("morie.data.cached_cpads", lambda: pd.DataFrame({"real": [1, 2]}))
    modules.run_module("descriptive-statistics")
    assert Path(seen["csv"]).name == "ocp21-cached.csv" and seen["frame"]["real"].tolist() == [1, 2]
    # nothing cached: the synthetic path goes through unchanged
    monkeypatch.setattr("morie.data.cached_cpads", lambda: None)
    modules.run_module("descriptive-statistics")
    assert seen["csv"] == str(modules.DEFAULT_CPADS_CSV)
