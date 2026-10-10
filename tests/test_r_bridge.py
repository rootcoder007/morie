"""The R bridge calls the real R function for every r_* command, with arguments passed as data."""

import shutil

import pytest

from morie import r_bridge as rb

needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")


def test_every_command_names_a_package_qualified_r_function():
    for name, spec in rb.SPECS.items():
        pkg, _, fn = spec.fn.partition("::")
        assert pkg and fn, name
        assert spec.recipe in {
            "call", "model", "aov_model", "svy", "boot", "plot", "ggplot", "survplot", "forestplot",
            "ipw", "aipw", "dml", "cssant", "missmap", "synth",
        }, name  # fmt: skip


def test_positional_arguments_take_the_r_names_from_the_usage():
    spec = rb.SPECS["r_power_t"]  # pwr.t.test(n, d, ...) takes d second; the usage gives it first
    assert rb.bind_arguments(spec, (0.5, None, 0.05, 0.8), {}) == [
        ("d", 0.5),
        ("n", None),
        ("sig.level", 0.05),
        ("power", 0.8),
    ]
    assert ("paired", True) in rb.bind_arguments(rb.SPECS["r_paired_t"], ([1], [2]), {})
    assert ("paired", False) in rb.bind_arguments(rb.SPECS["r_paired_t"], ([1], [2]), {"paired": False})
    with pytest.raises(rb.RBridgeError, match="given twice"):
        rb.bind_arguments(rb.SPECS["r_ttest"], ([1.0],), {"x": [2.0]})


@pytest.mark.parametrize(
    "formula",
    ["y ~ x1 + x2", "y ~ .", "Surv(time, status) ~ g", "y ~ s(x1) + log(x2)", "y ~ x + (1 | g)", "y ~ x1 | z",
     "~ 1 | g", "y ~ I(x^2) + x:z - 1", "y ~ x %in% g"],
)  # fmt: skip
def test_ordinary_formulas_pass(formula):
    assert rb.check_formula(formula) == formula


@pytest.mark.parametrize(
    "formula",
    ['y ~ system("id")', "y ~ x; q()", "y ~ eval(x)", "y ~ `x`", "y ~ x$a", 'y ~ "a"', "y ~ {x}", "y ~ x <- 1",
     "y ~ repeat x", "y ~ function(x) x", "y ~ base::system(x)"],
)  # fmt: skip
def test_formulas_that_could_run_code_are_refused(formula):
    with pytest.raises(rb.RBridgeError):
        rb.check_formula(formula)


def test_prompt_words_become_values(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("y,g\n1,a\n2,b\n")
    args, kwargs = rb.parse_words(["y ~ g", str(csv), "1,2,3", "0.5", "true", "method=BH", "two.sided"])
    assert args[0] == "y ~ g"
    assert list(args[1].columns) == ["y", "g"]
    assert args[2:] == [[1, 2, 3], 0.5, True, "two.sided"]
    assert kwargs == {"method": "BH"}


@needs_r
def test_r_commands_give_the_numbers_of_the_r_function():
    # morie's native when one is installed (t = 1.45048, df = 4, p = 0.2205, as stats::t.test), else R's
    out = rb.call("r_ttest", [5.1, 4.9, 5.6, 5.0, 5.3], 5)
    assert "1.450" in out and "0.2205" in out
    out = rb.call("r_paired_t", [1.0, 2.0, 3.0, 4.0], [1.5, 2.1, 3.9, 4.2])
    assert "aired" in out and "-2.365" in out  # t = -2.3651 either way
    assert "0.04" in rb.call("r_p_adjust", [0.01, 0.02, 0.04, 0.3], "BH")


@needs_r
def test_a_data_frame_and_formula_reach_r_as_data():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"y": [1.0, 2.1, 2.9, 4.2, 5.1, 5.8], "x": [1, 2, 3, 4, 5, 6], "g": list("aabbcc")})
    out = rb.call("r_lm", "y ~ x", df)
    assert "0.98" in out and "0.0866" in out  # slope 0.98, intercept 0.08667, as stats::lm and numpy
    assert "5.8" not in out  # the data's values are not echoed
    assert "[1] 6 3" in rb.call("r_dim", df)
    assert "Factor w/ 3 levels" in rb.call("r_str", df)  # text columns arrive as factors


@needs_r
def test_arguments_never_run_as_r_code():
    out = rb.call("r_summary", 'x"); print("INJECTED')
    assert "INJECTED" not in out and "character" in out
    assert "numeric" in rb.call("r_class", 3.5)
    assert "character" in rb.call("r_class", "3.5")
    with pytest.raises(rb.RBridgeError, match="not allowed in a formula"):
        rb.call("r_lm", 'y ~ system("echo INJECTED")', {"y": [1.0]})


@needs_r
def test_r_errors_and_missing_packages_are_reported():
    with pytest.raises(rb.RBridgeError, match="r_ttest"):
        rb.call("r_ttest", [1.0])  # t.test needs more than one value
    with pytest.raises(rb.RBridgeError, match="unknown R bridge command"):
        rb.call("r_nope")


def test_the_registry_uses_the_bridge():
    from morie import stat_commands

    stat_commands._register_r_bridge()
    cmd = stat_commands.COMMAND_REGISTRY["r_paired_t"]
    assert cmd.is_r_bridge and cmd.description == "morie native, checked against R t.test(paired = TRUE)"
    assert stat_commands.COMMAND_REGISTRY["r_levene"].description == "morie native, checked against R car::leveneTest()"
    assert stat_commands.COMMAND_REGISTRY["r_head"].description == "R head()"
