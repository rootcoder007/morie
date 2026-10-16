import io
from pathlib import Path

import pytest

from morie.runner import build_parser, execute_pipeline, main

_RTESTS_DIR = Path(__file__).resolve().parents[1] / "rtests"


class _TtyStdin(io.StringIO):
    """stdin that reports a terminal, so the confirmation prompt (mocked below) is reached under pytest."""

    def isatty(self):
        return True


def test_execute_pipeline_requires_confirmation_by_default(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _TtyStdin())
    monkeypatch.setattr("builtins.input", lambda _: "n")

    status = execute_pipeline()

    captured = capsys.readouterr()
    assert status == 1
    assert "Selected modules:" in captured.out
    assert "Pipeline aborted." in captured.out


def test_execute_pipeline_runs_when_confirmed(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _TtyStdin())
    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr("morie.runner.run_module", lambda *_, **__: {})

    status = execute_pipeline(modules=["power-design"], silent=False)

    captured = capsys.readouterr()
    assert status == 0
    assert "Selected modules: power-design" in captured.out
    assert "Pipeline completed successfully (" in captured.out
    assert "Completed modules: power-design" in captured.out


def test_main_prints_help_when_no_action_requested(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["morie"])

    status = main()

    captured = capsys.readouterr()
    assert status == 0
    assert "MORIE" in captured.out


def test_parser_accepts_module_overrides():
    parser = build_parser()
    args = parser.parse_args(["pipeline", "--all", "--modules", "power-design", "propensity-scores", "-y"])

    assert args.all is True
    assert args.modules == ["power-design", "propensity-scores"]
    assert args.yes is True


def test_parser_accepts_run_module_arguments():
    parser = build_parser()
    args = parser.parse_args(
        ["run-module", "power-design", "--cpads-csv", "docs/source/datasets/cpads-2021-2022-pumf2.csv"]
    )

    assert args.module == "power-design"
    assert args.cpads_csv.endswith("cpads-2021-2022-pumf2.csv")


@pytest.mark.skipif(not _RTESTS_DIR.exists(), reason="needs the sibling rtests/ legacy root")
def test_main_runs_parity_review(monkeypatch, capsys, tmp_path):
    csv_path = tmp_path / "parity.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "morie",
            "parity-review",
            "--epiml-root",
            str(_RTESTS_DIR),
            "--output",
            str(csv_path),
        ],
    )

    status = main()

    captured = capsys.readouterr()
    assert status == 0
    assert "Audit summary:" in captured.out or "Parity summary:" in captured.out
    assert csv_path.exists()


def test_main_lists_modules(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["morie", "list-modules"])
    status = main()
    captured = capsys.readouterr()
    assert status == 0
    assert "power-design" in captured.out


def test_models_verb_lists_the_route_models(monkeypatch, capsys):
    from morie import llm, providers

    monkeypatch.setenv("GROQ_API_KEY", "gsk-1")
    monkeypatch.setenv("MORIE_GROQ_MODEL", "llama-b")
    monkeypatch.setenv("OLLAMA_HOST", "off")
    monkeypatch.delenv("MORIE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.setattr(providers, "list_models", lambda pid, timeout=10.0: ["llama-a", "llama-b"])
    llm._reset_route_cache()
    monkeypatch.setattr("sys.argv", ["morie", "models"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "Groq; the model in use is marked *:" in out and "  * llama-b" in out and "    llama-a" in out
    monkeypatch.setattr("sys.argv", ["morie", "models", "--route", "hosted"])
    assert main() == 2
    assert "retired" in capsys.readouterr().out
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setattr("sys.argv", ["morie", "models"])
    assert main() == 1
    assert "ask has no route" in capsys.readouterr().out


def test_main_impl_does_not_shadow_path():
    """A local `from pathlib import Path` inside _main_impl made every earlier
    Path(...) in the function raise UnboundLocalError (morie pull --out, 1.3.5)."""
    from morie import runner

    assert "Path" not in runner._main_impl.__code__.co_varnames
