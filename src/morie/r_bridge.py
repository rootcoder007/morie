"""R bridge: call R functions from morie, with every argument passed to R as data.

Each ``r_*`` command names a real R function (``stats::t.test``, ``car::leveneTest``, ...) and the
R names of the arguments in its documented usage. Arguments are written to files in a private
temporary directory and read back by the fixed script ``_R_SCRIPT``: numbers, text, logicals,
data frames, matrices and lists arrive in R as values, never as source. The one thing R has to
parse is a model formula, so formula strings are checked against an allow-list of names,
operators and functions (``log``, ``s``, ``Surv``, ...) before they are written.
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

_NATIVES_R = Path(__file__).with_name("rscripts") / "bridge_natives.R"


def native_commands() -> frozenset[str]:
    """The r_* commands that run morie's own R functions (adapters in rscripts/bridge_natives.R)."""
    try:
        text = _NATIVES_R.read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    return frozenset(re.findall(r'^NATIVE\[\["(r_\w+)"\]\]', text, re.M))


class RBridgeError(RuntimeError):
    """R failed, is missing, or was given an argument the bridge will not pass on."""


@dataclass(frozen=True)
class RSpec:
    """How one ``r_*`` command calls R."""

    fn: str  # "pkg::name"
    params: tuple[str, ...]  # R names of the documented positional arguments, in usage order; "*" = by position
    usage: str
    fixed: Mapping[str, Any] = field(default_factory=dict)  # added unless the caller sets them
    recipe: str = "call"  # how the R script uses the function; see _R_SCRIPT
    label: str = ""  # description shown in the command list; derived from fn when empty


def _s(fn: str, params: str, usage: str, recipe: str = "call", label: str = "", **fixed: Any) -> RSpec:
    return RSpec(fn, tuple(params.split()), usage, fixed, recipe, label)


# "@morie::" names a function in morie's own R package: rmorie or morie, whichever matches this
# morie's version (the rule the R-backed modules use, see morie.modules._r_route_ready).
MORIE_R = "@morie"


SPECS: dict[str, RSpec] = {
    # Statistical tests
    "r_ttest": _s("stats::t.test", "x mu", "r_ttest(x, mu)"),
    "r_ttest2": _s("stats::t.test", "x y", "r_ttest2(x, y)"),
    "r_paired_t": _s("stats::t.test", "x y", "r_paired_t(x, y)", paired=True),
    "r_wilcox": _s("stats::wilcox.test", "x y", "r_wilcox(x, y)"),
    "r_anova": _s("stats::aov", "formula data", "r_anova(formula, data)"),
    "r_kruskal": _s("stats::kruskal.test", "formula data", "r_kruskal(formula, data)"),
    "r_chisq": _s("stats::chisq.test", "x y", "r_chisq(x, y)"),
    "r_fisher": _s("stats::fisher.test", "x y", "r_fisher(x, y)"),
    "r_shapiro": _s("stats::shapiro.test", "x", "r_shapiro(x)"),
    "r_ks": _s("stats::ks.test", "x y", "r_ks(x, y)"),
    "r_cor": _s("stats::cor.test", "x y method", "r_cor(x, y, method)"),
    "r_mcnemar": _s("stats::mcnemar.test", "x y", "r_mcnemar(x)"),
    "r_bartlett": _s("stats::bartlett.test", "formula data", "r_bartlett(formula, data)"),
    "r_levene": _s("car::leveneTest", "y data", "r_levene(formula, data)"),
    "r_friedman": _s("stats::friedman.test", "* data", "r_friedman(formula, data) or r_friedman(matrix)"),
    # Regression
    "r_lm": _s("stats::lm", "formula data", "r_lm(formula, data)"),
    "r_glm": _s("stats::glm", "formula data family", "r_glm(formula, data, family)"),
    "r_logistic": _s("stats::glm", "formula data", "r_logistic(formula, data)", family="binomial"),
    "r_poisson": _s("stats::glm", "formula data", "r_poisson(formula, data)", family="poisson"),
    "r_nls": _s("stats::nls", "formula data start", "r_nls(formula, data, start)"),
    "r_lme": _s("nlme::lme", "fixed data random", "r_lme(formula, data, random)"),
    "r_lmer": _s("lme4::lmer", "formula data", "r_lmer(formula, data)"),
    "r_glmer": _s("lme4::glmer", "formula data family", "r_glmer(formula, data, family)"),
    "r_gam": _s("mgcv::gam", "formula data", "r_gam(formula, data)"),
    "r_quantreg": _s("quantreg::rq", "formula data tau", "r_quantreg(formula, data, tau)"),
    "r_robust": _s("MASS::rlm", "formula data", "r_robust(formula, data)"),
    # Survival
    "r_surv": _s("survival::Surv", "time event", "r_surv(time, event)"),
    "r_survfit": _s("survival::survfit", "formula data", "r_survfit(formula, data)"),
    "r_coxph": _s("survival::coxph", "formula data", "r_coxph(formula, data)"),
    "r_survdiff": _s("survival::survdiff", "formula data", "r_survdiff(formula, data)"),
    "r_aft": _s("survival::survreg", "formula data dist", "r_aft(formula, data, dist)"),
    # Causal
    "r_matchit": _s("MatchIt::matchit", "formula data method", "r_matchit(formula, data, method)"),
    "r_ipw": _s("ipw::ipwpoint", "formula data", "r_ipw(formula, data)", recipe="ipw"),
    "r_aipw": _s("AIPW::AIPW", "y a w", "r_aipw(y, a, w)", recipe="aipw"),
    "r_dml": _s("DoubleML::DoubleMLPLR", "data y d x", "r_dml(data, y, d, x)", recipe="dml"),
    "r_irm": _s("DoubleML::DoubleMLIRM", "data y d x", "r_irm(data, y, d, x)", recipe="dml"),
    "r_didR": _s(
        "@morie::morie_cssant",
        "formula data gname tname",
        "r_didR(formula, data, gname, tname, idname=)",
        "cssant",
        "morie Callaway-Sant'Anna DiD (morie_cssant)",
    ),
    "r_rdrobust": _s("rdrobust::rdrobust", "y x c", "r_rdrobust(y, x, c)"),
    "r_ivreg": _s("ivreg::ivreg", "formula data", "r_ivreg(formula, data)"),
    "r_synth": _s("Synth::synth", "data", "r_synth(data, **dataprep_args)", recipe="synth"),
    # Multiple testing
    "r_p_adjust": _s("stats::p.adjust", "p method", "r_p_adjust(p, method)"),
    # Diagnostics: fit lm(formula, data), then run the diagnostic on the model
    "r_vif": _s("car::vif", "formula data", "r_vif(formula, data)", recipe="model"),
    "r_durbinwatson": _s("car::durbinWatsonTest", "formula data", "r_durbinwatson(formula, data)", recipe="model"),
    "r_bptest": _s("lmtest::bptest", "formula data", "r_bptest(formula, data)", recipe="model"),
    "r_resettest": _s("lmtest::resettest", "formula data", "r_resettest(formula, data)", recipe="model"),
    # Effect sizes
    "r_cohens_d": _s("effectsize::cohens_d", "x y", "r_cohens_d(x, y)"),
    "r_hedges_g": _s("effectsize::hedges_g", "x y", "r_hedges_g(x, y)"),
    "r_eta_sq": _s("effectsize::eta_squared", "formula data", "r_eta_sq(formula, data)", recipe="aov_model"),
    "r_cramers_v": _s("effectsize::cramers_v", "x y", "r_cramers_v(x, y)"),
    # Power
    "r_power_t": _s("pwr::pwr.t.test", "d n sig.level power", "r_power_t(d, n, sig, power)"),
    "r_power_anova": _s("pwr::pwr.anova.test", "k n f sig.level", "r_power_anova(k, n, f, sig)"),
    "r_power_chisq": _s("pwr::pwr.chisq.test", "w N df sig.level", "r_power_chisq(w, N, df, sig)"),
    "r_power_prop": _s("pwr::pwr.2p.test", "h n sig.level", "r_power_prop(h, n, sig)"),
    # Tables
    "r_table1": _s("tableone::CreateTableOne", "data vars strata", "r_table1(data, vars, strata)"),
    "r_gtsummary": _s(
        "@morie::table1", "data group_col", "r_gtsummary(data, by)", "call", "morie table1() summary table"
    ),
    "r_stargazer": _s("stargazer::stargazer", "formula data", "r_stargazer(formula, data)", "model", type="text"),
    # Missing data
    "r_mice": _s("mice::mice", "data m method", "r_mice(data, m, method)", printFlag=False),
    "r_amelia": _s("Amelia::amelia", "x m", "r_amelia(data, m)", p2s=0),
    "r_naniar": _s("graphics::image", "data", "r_naniar(data, file=)", "missmap", "R missingness map (base graphics)"),
    # Visualization: saved as PNG; the path is printed
    "r_ggplot": _s("ggplot2::ggplot", "data x y geom", "r_ggplot(data, x, y, geom, file=)", recipe="ggplot"),
    "r_ggsurvplot": _s("survminer::ggsurvplot", "formula data", "r_ggsurvplot(formula, data, file=)", "survplot"),
    "r_forestplot": _s("forestplot::forestplot", "data", "r_forestplot(data, file=)", recipe="forestplot"),
    # Survey weights: data plus ids=, weights=, strata= describe the design
    "r_svydesign": _s("survey::svydesign", "ids weights data", "r_svydesign(ids, weights, data)"),
    "r_svymean": _s("survey::svymean", "x data", "r_svymean(formula, data, weights=)", recipe="svy"),
    "r_svytotal": _s("survey::svytotal", "x data", "r_svytotal(formula, data, weights=)", recipe="svy"),
    "r_svyglm": _s("survey::svyglm", "formula data family", "r_svyglm(formula, data, family, weights=)", "svy"),
    "r_calibrate": _s("survey::calibrate", "data formula population", "r_calibrate(data, formula, pop)", "svy"),
    "r_rake": _s("survey::rake", "data sample.margins population.margins", "r_rake(data, sample, pop)", "svy"),
    # Bootstrap: statistic is a name (mean, median, sd, var, ...)
    "r_boot": _s("boot::boot", "data statistic R", "r_boot(data, statistic, R)", recipe="boot"),
    "r_boot_ci": _s("boot::boot.ci", "data statistic R type", "r_boot_ci(data, statistic, R, type)", recipe="boot"),
    # Meta-analysis
    "r_metafor": _s("metafor::rma", "yi vi method", "r_metafor(yi, vi, method)"),
    "r_meta_bin": _s("meta::metabin", "event.e n.e event.c n.c", "r_meta_bin(event_e, n_e, event_c, n_c)"),
    # Misc
    "r_summary": _s("base::summary", "object", "r_summary(obj)"),
    "r_str": _s("utils::str", "object", "r_str(obj)"),
    "r_head": _s("utils::head", "x n", "r_head(data, n)"),
    "r_tail": _s("utils::tail", "x n", "r_tail(data, n)"),
    "r_dim": _s("base::dim", "x", "r_dim(data)"),
    "r_names": _s("base::names", "x", "r_names(data)"),
    "r_class": _s("base::class", "x", "r_class(obj)"),
}


# ---------------------------------------------------------------------------
# Formulas: the only argument R parses, so only names, numbers, operators and the
# functions below may appear in one.
# ---------------------------------------------------------------------------

FORMULA_FUNCTIONS = frozenset(
    {
        "I", "log", "log1p", "log2", "log10", "exp", "sqrt", "abs", "scale", "factor", "as.factor",
        "as.numeric", "as.integer", "poly", "ns", "bs", "s", "te", "ti", "strata", "Surv", "cluster",
        "offset", "interaction",
    }
)  # fmt: skip
_RESERVED = frozenset({"if", "else", "repeat", "while", "function", "for", "next", "break", "in", "return"})
_FORMULA_TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<name>[A-Za-z.][A-Za-z0-9._]*)|(?P<num>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"
    r"|(?P<op>%in%|[~+\-*/:^|(),])"
)


def check_formula(text: str) -> str:
    """Return ``text`` if it is a formula the bridge will pass to R, else raise RBridgeError."""
    if len(text) > 2000 or "~" not in text:
        raise RBridgeError(f"not a formula: {text!r}")
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _FORMULA_TOKEN.match(text, pos)
        if m is None:
            raise RBridgeError(f"formula {text!r}: {text[pos]!r} is not allowed in a formula")
        if m.lastgroup != "space":
            tokens.append((m.lastgroup or "", m.group()))
        pos = m.end()
    for i, (kind, tok) in enumerate(tokens):
        if kind != "name":
            continue
        if tok in _RESERVED:
            raise RBridgeError(f"formula {text!r}: {tok!r} is not allowed in a formula")
        called = i + 1 < len(tokens) and tokens[i + 1][1] == "("
        if called and tok not in FORMULA_FUNCTIONS:
            raise RBridgeError(
                f"formula {text!r}: the function {tok}() is not allowed in a formula; "
                f"allowed: {', '.join(sorted(FORMULA_FUNCTIONS))}"
            )
    return text


# ---------------------------------------------------------------------------
# Writing arguments as data
# ---------------------------------------------------------------------------

_R_NAME = re.compile(r"^[A-Za-z.][A-Za-z0-9._]*$")


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int | float):
        return True
    return type(v).__module__ == "numpy" and hasattr(v, "item") and isinstance(v.item(), int | float)


def _num_text(v: Any) -> str:
    if v is None:
        return "NA"
    f = float(v.item() if hasattr(v, "item") else v)
    if math.isnan(f):
        return "NA"
    if math.isinf(f):
        return "Inf" if f > 0 else "-Inf"
    return repr(int(f)) if isinstance(v, int) and not isinstance(v, bool) else repr(f)


def _chr_text(v: Any) -> str:
    return "%NA%" if v is None else quote(str(v), safe="")


def _is_frame(v: Any) -> bool:
    return hasattr(v, "to_csv") and hasattr(v, "columns") and not isinstance(v, type)


def _as_list(v: Any) -> list | None:
    if isinstance(v, list | tuple):
        return list(v)
    if hasattr(v, "tolist") and not isinstance(v, str | bytes):
        out = v.tolist()
        return out if isinstance(out, list) else [out]
    return None


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


class _ArgWriter:
    """Writes arguments into a directory as files plus an index R reads with read.csv."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, value: Any) -> None:
        if name and not _R_NAME.match(name):
            raise RBridgeError(f"{name!r} is not a valid R argument name")
        file = f"a{len(self.rows)}"
        kind = self._write(self.dir / file, value)
        self.rows.append((name, kind, file))

    def close(self) -> None:
        with (self.dir / "index.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "type", "file"])
            w.writerows(self.rows)

    def _write(self, path: Path, v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            _write_lines(path, ["TRUE" if v else "FALSE"])
            return "lgl"
        if _is_number(v):
            _write_lines(path, [_num_text(v)])
            return "num"
        if isinstance(v, str):
            if "~" in v:
                _write_lines(path, [_chr_text(check_formula(v))])
                return "formula"
            _write_lines(path, [_chr_text(v)])
            return "chr"
        if _is_frame(v):
            v.to_csv(str(path), index=False)
            return "df"
        if isinstance(v, Mapping):
            if v and all(_is_number(x) or x is None for x in v.values()):
                self._check_names(v)
                _write_lines(path, [f"{_chr_text(k)},{_num_text(x)}" for k, x in v.items()])
                return "nnum"
            sub = _ArgWriter(path)
            for k, x in v.items():
                sub.add(str(k), x)
            sub.close()
            return "list"
        items = _as_list(v)
        if items is None:
            raise RBridgeError(f"cannot pass a {type(v).__name__} to R")
        return self._write_vector(path, items)

    def _write_vector(self, path: Path, items: list) -> str:
        present = [x for x in items if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if all(_is_number(x) for x in present):
            _write_lines(path, [_num_text(x) for x in items])
            return "num"
        if all(isinstance(x, bool) for x in present):
            _write_lines(path, ["NA" if x is None else ("TRUE" if x else "FALSE") for x in items])
            return "lgl"
        if all(isinstance(x, str) for x in present) and not any("~" in x for x in present):
            _write_lines(path, [_chr_text(x) for x in items])
            return "chr"
        rows = [_as_list(x) for x in items]
        if items and all(r is not None and all(_is_number(c) or c is None for c in r) for r in rows):
            if len({len(r) for r in rows if r is not None}) != 1:
                raise RBridgeError("a matrix needs rows of equal length")
            _write_lines(path, [",".join(_num_text(c) for c in r) for r in rows if r is not None])
            return "matrix"
        sub = _ArgWriter(path)
        for x in items:
            sub.add("", x)
        sub.close()
        return "list"

    @staticmethod
    def _check_names(v: Mapping) -> None:
        for k in v:
            if not isinstance(k, str):
                raise RBridgeError(f"names must be text, not {k!r}")


# ---------------------------------------------------------------------------
# The R side. A constant: the function comes from SPECS and is resolved with getExportedValue,
# every argument is read from a file as a value, and formulas were checked before writing.
# ---------------------------------------------------------------------------

_R_SCRIPT = r"""
`%||%` <- function(a, b) if (is.null(a)) b else a
fail <- function(msg) { cat(msg, "\n", file = stderr()); quit(save = "no", status = 1) }
dec <- function(x) { out <- vapply(x, utils::URLdecode, "", USE.NAMES = FALSE); out[x == "%NA%"] <- NA
  Encoding(out) <- "UTF-8"; out }
lines <- function(p) readLines(p, warn = FALSE, encoding = "UTF-8")
read_arg <- function(p, type) switch(type,
  null = NULL,
  num = as.numeric(lines(p)),
  lgl = as.logical(lines(p)),
  chr = dec(lines(p)),
  formula = stats::as.formula(dec(lines(p)), env = globalenv()),
  df = utils::read.csv(p, check.names = FALSE, stringsAsFactors = TRUE),
  matrix = unname(as.matrix(utils::read.csv(p, header = FALSE))),
  nnum = { x <- utils::read.csv(p, header = FALSE, colClasses = c("character", "numeric"))
    stats::setNames(x[[2]], dec(x[[1]])) },
  list = read_args(p),
  fail(paste("unknown argument type", type)))
read_args <- function(d) {
  idx <- utils::read.csv(file.path(d, "index.csv"), colClasses = "character", na.strings = character(0))
  out <- lapply(seq_len(nrow(idx)), function(i) read_arg(file.path(d, idx$file[i]), idx$type[i]))
  if (any(nzchar(idx$name))) names(out) <- idx$name
  out
}
get_fun <- function(spec) {
  p <- strsplit(spec, "::", fixed = TRUE)[[1]]
  if (!requireNamespace(p[1], quietly = TRUE))
    fail(sprintf("the R package '%s' is not installed; install it in R with install.packages(\"%s\")", p[1], p[1]))
  if (!p[1] %in% c("base", "stats", "utils"))
    suppressPackageStartupMessages(library(p[1], character.only = TRUE))
  getExportedValue(p[1], p[2])
}
# Call f with each long argument bound to a name in a fresh environment, so a model's printed call
# reads lm(formula = y ~ x, data = data) rather than repeating every value of the data.
invoke <- function(spec, f, a) {
  e <- new.env(parent = globalenv())
  fname <- strsplit(spec, "::", fixed = TRUE)[[1]][2]
  assign(fname, f, envir = e)
  nms <- names(a) %||% rep("", length(a))
  args <- lapply(seq_along(a), function(i) {
    x <- a[[i]]
    if (is.null(x) || inherits(x, "formula") || (is.atomic(x) && length(x) <= 1)) return(x)
    sym <- if (nzchar(nms[i])) nms[i] else paste0("arg", i)
    assign(sym, x, envir = e)
    as.name(sym)
  })
  names(args) <- nms
  do.call(fname, args, envir = e)
}
drop <- function(a, nms) a[setdiff(names(a) %||% character(0), nms)]
rhs_formula <- function(f) { if (length(f) == 3) f[[2]] <- NULL; f }
stat_fun <- function(name) {
  ok <- c("mean", "median", "sd", "var", "min", "max", "IQR", "mad", "sum")
  if (!is.character(name) || !name %in% ok) fail(paste("statistic must be one of:", paste(ok, collapse = ", ")))
  f <- get(name, envir = globalenv(), mode = "function")
  function(d, i) f(d[i])
}
save_plot <- function(a, draw) {
  file <- a$file %||% tempfile("morie-plot-", fileext = ".png")
  grDevices::png(file, width = 1600, height = 1000, res = 150)
  draw()
  grDevices::dev.off()
  cat("plot saved to", file, "\n")
  invisible(NULL)
}
design_of <- function(a) {
  get_fun("survey::svydesign")
  survey::svydesign(ids = a$ids %||% ~1, weights = a$weights, strata = a$strata, data = a$data)
}

d <- commandArgs(trailingOnly = TRUE)[1]
call <- lines(file.path(d, "call.txt"))
spec <- call[1]; recipe <- call[2]
if (startsWith(spec, "@morie::")) spec <- sub("^@morie", call[3], spec)
if (identical(recipe, "native")) {
  # morie's own R functions, through the adapter for this command (rscripts/bridge_natives.R)
  if (!requireNamespace(call[3], quietly = TRUE)) fail(sprintf("the R package '%s' is not installed", call[3]))
  natives <- new.env(parent = globalenv())
  sys.source(call[5], envir = natives)
  args <- read_args(file.path(d, "args"))
  res <- tryCatch(natives$NATIVE[[call[4]]](args, asNamespace(call[3])), error = function(e) fail(conditionMessage(e)))
  if (is.list(res) && is.null(oldClass(res)) && length(names(res))) {
    # a plain list: one line per short value, a heading above each table or longer value
    for (n in names(res)) {
      v <- res[[n]]
      if (is.null(v) || is.function(v) || is.environment(v)) next
      if (is.atomic(v) && length(v) == 1L && is.null(dim(v))) cat(sprintf("%-18s %s\n", n, format(v)))
      else { cat("\n", n, ":\n", sep = ""); print(v) }
    }
  } else if (!is.null(res)) print(res)
  quit(save = "no", status = 0)
}
a <- read_args(file.path(d, "args"))
models <- c("lm", "glm", "aov", "nls", "lme", "merMod", "gam", "rq", "rlm", "coxph", "survreg",
            "ivreg", "matchit", "rdrobust", "svyglm")

res <- tryCatch({
  f <- get_fun(spec)
  switch(recipe,
    call = invoke(spec, f, a),
    model = , aov_model = {
      fit <- if (recipe == "aov_model") "stats::aov" else "stats::lm"
      m <- invoke(fit, get_fun(fit), list(formula = a$formula, data = a$data))
      invoke(spec, f, c(list(m), drop(a, c("formula", "data"))))
    },
    svy = invoke(spec, f, c(drop(a, c("ids", "weights", "strata", "data")), list(design = design_of(a)))),
    boot = {
      b <- boot::boot(a$data, stat_fun(a$statistic), R = a$R %||% 1000)
      if (spec == "boot::boot.ci") boot::boot.ci(b, type = a$type %||% "perc") else b
    },
    plot = save_plot(a, function() print(do.call(f, drop(a, "file")))),
    ggplot = save_plot(a, function() {
      geoms <- c("point", "line", "col", "bar", "histogram", "boxplot", "density", "smooth")
      g <- a$geom %||% "point"
      if (!g %in% geoms) fail(paste("geom must be one of:", paste(geoms, collapse = ", ")))
      map <- if (is.null(a$y)) ggplot2::aes(x = .data[[a$x]]) else ggplot2::aes(x = .data[[a$x]], y = .data[[a$y]])
      print(f(a$data, map) + getExportedValue("ggplot2", paste0("geom_", g))())
    }),
    survplot = save_plot(a, function() {
      get_fun("survival::survfit")
      fit <- do.call(survival::survfit, list(formula = a$formula, data = a$data))
      print(f(fit, data = a$data))
    }),
    forestplot = save_plot(a, function() {
      x <- a$data
      print(f(labeltext = as.character(x[[a$label %||% "label"]]), mean = x[[a$mean %||% "mean"]],
              lower = x[[a$lower %||% "lower"]], upper = x[[a$upper %||% "upper"]]))
    }),
    ipw = {
      w <- do.call(f, list(exposure = a$formula[[2]], family = a$family %||% "binomial",
                           link = a$link %||% "logit", denominator = rhs_formula(a$formula), data = a$data))
      print(summary(w$ipw.weights)); w$ipw.weights
    },
    aipw = {
      get_fun("SuperLearner::SuperLearner")  # AIPW needs SuperLearner attached for its learner names
      obj <- f$new(Y = a$y, A = a$a, W = a$w, Q.SL.library = a$Q.SL.library %||% "SL.glm",
                   g.SL.library = a$g.SL.library %||% "SL.glm", k_split = a$k_split %||% 10, verbose = FALSE)
      obj$fit(); obj$summary(); obj$result
    },
    dml = {
      get_fun("mlr3::lrn")
      dat <- DoubleML::DoubleMLData$new(a$data, y_col = a$y, d_cols = a$d, x_cols = a$x)
      ml_l <- mlr3::lrn(a$ml_l %||% "regr.rpart")
      ml_m <- mlr3::lrn(a$ml_m %||% if (spec == "DoubleML::DoubleMLIRM") "classif.rpart" else "regr.rpart")
      obj <- f$new(dat, ml_l, ml_m); obj$fit(); obj$summary(); invisible(obj)
    },
    cssant = {
      if (!identical(all.vars(rhs_formula(a$formula)), character(0)))
        fail("morie_cssant takes no covariates: give the formula as y ~ 1")
      x <- a$data
      need <- c(all.vars(a$formula[[2]]), a$idname %||% fail("give idname= (the unit identifier column)"),
                a$tname, a$gname)
      miss <- setdiff(need, names(x))
      if (length(miss)) fail(paste("not in the data:", paste(miss, collapse = ", ")))
      g <- x[[a$gname]]; tt <- x[[a$tname]]
      D <- as.integer(!is.na(g) & g > 0 & tt >= g)  # did's convention: gname 0 = never treated
      r <- f(y = x[[need[1]]], D = D, unit = x[[a$idname]], time = tt, control = a$control %||% "notyet")
      cat(sprintf("Callaway-Sant'Anna ATT (morie_cssant, control = %s)\n", r$control_group %||% a$control %||% "notyet"))
      cat(sprintf("overall ATT %.4f  SE %.4f  95%% CI [%.4f, %.4f]\n", r$estimate, r$se, r$ci[1], r$ci[2]))
      gt <- strsplit(names(r$att_gt), "|", fixed = TRUE)
      cat("\ngroup-time ATT(g, t), cohort and period as morie_cssant indexes them:\n")
      print(data.frame(cohort = vapply(gt, `[`, "", 1), period = vapply(gt, `[`, "", 2),
                       att = round(unlist(r$att_gt, use.names = FALSE), 4)), row.names = FALSE)
      if (length(r$event)) {
        ev <- do.call(rbind, r$event)
        cat("\nevent-study aggregation:\n")
        print(data.frame(event_time = names(r$event), att = round(ev[, 1], 4), se = round(ev[, 2], 4)),
              row.names = FALSE)
      }
      if (!is.null(r$pretrend_note)) cat("\n", r$pretrend_note, "\n", sep = "")
      invisible(NULL)
    },
    missmap = {
      x <- as.data.frame(a$data); m <- is.na(x)
      tab <- data.frame(variable = names(x), n_missing = colSums(m), pct_missing = round(100 * colMeans(m), 1),
                        row.names = NULL)
      print(tab)
      cat(sprintf("\n%.1f%% of all values are missing\n", 100 * mean(m)))
      save_plot(a, function() {
        op <- graphics::par(mar = c(8, 4, 3, 1)); on.exit(graphics::par(op))
        f(seq_len(ncol(m)), seq_len(nrow(m)), t(m[rev(seq_len(nrow(m))), , drop = FALSE]) * 1,
          col = c("grey88", "grey20"), zlim = c(0, 1), axes = FALSE, xlab = "", ylab = "observations",
          main = sprintf("Missing values (dark): %.1f%% overall", 100 * mean(m)))
        graphics::axis(1, at = seq_len(ncol(m)), labels = sprintf("%s (%.1f%%)", names(x), tab$pct_missing),
                       las = 2, cex.axis = 0.8)
        graphics::box()
      })
    },
    synth = {
      x <- a$data
      x[] <- lapply(x, function(v) if (is.factor(v)) as.character(v) else v)  # dataprep wants text, not factors
      prep <- do.call(Synth::dataprep, c(list(foo = x), drop(a, "data")))
      s <- f(prep); print(Synth::synth.tab(dataprep.res = prep, synth.res = s)); invisible(s)
    },
    fail(paste("unknown recipe", recipe)))
}, error = function(e) fail(conditionMessage(e)))

if (inherits(res, "gtsummary")) res <- gtsummary::as_kable(res)
if (!is.null(res) && !identical(spec, "stargazer::stargazer")) {
  if (inherits(res, models)) print(summary(res)) else print(res)
}
"""


def rscript() -> str | None:
    """Path of Rscript, or None."""
    return shutil.which("Rscript")


def _morie_r_package() -> str:
    """rmorie or morie: the R package whose version matches this morie, by the R-backed modules' rule."""
    from morie import modules

    try:
        modules._r_route_ready()
    except RuntimeError as e:
        raise RBridgeError(str(e)) from e
    return str(modules._R_PACKAGE)


def bind_arguments(spec: RSpec, args: tuple, kwargs: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Name positional arguments with the R names in ``spec.params``, then add keywords and fixed values."""
    out: list[tuple[str, Any]] = []
    named: dict[str, Any] = {}
    first: list[tuple[str, Any]] = []
    for i, v in enumerate(args):
        if i < len(spec.params) and spec.params[i] != "*":
            named[spec.params[i]] = v
        elif i == 0:
            first.append(("", v))  # "*": passed by position, so an S3 generic dispatches on it
        else:
            out.append(("", v))
    for k, v in kwargs.items():
        if k in named:
            raise RBridgeError(f"{k} given twice")
        named[k] = v
    for k, v in spec.fixed.items():
        named.setdefault(k, v)
    return first + [(k, v) for k, v in named.items()] + out


def call(command: str, *args: Any, timeout: float | None = None, **kwargs: Any) -> str:
    """Run the R function behind ``command`` and return what R printed.

    Raises RBridgeError when R is missing, a package is not installed, an argument cannot be passed,
    or the R call fails; the message is R's own error.
    """
    spec = SPECS.get(command)
    if spec is None:
        raise RBridgeError(f"unknown R bridge command: {command}")
    exe = rscript()
    if exe is None:
        raise RBridgeError("R not found. Install R to use R bridge commands.")
    from morie._interactive import LayerMissingError, launcher

    try:
        sp = launcher("The R bridge")
    except LayerMissingError as e:
        raise RBridgeError(str(e)) from e
    if timeout is None:
        timeout = float(os.environ.get("MORIE_R_TIMEOUT", "300"))
    with tempfile.TemporaryDirectory(prefix="morie-r-") as tmp:
        root = Path(tmp)
        native = command in native_commands()
        pkg = _morie_r_package() if native or spec.fn.startswith(MORIE_R) else ""
        _write_lines(root / "call.txt", [spec.fn, "native" if native else spec.recipe, pkg, command, str(_NATIVES_R)])
        writer = _ArgWriter(root / "args")
        for name, value in bind_arguments(spec, args, kwargs):
            writer.add(name, value)
        writer.close()
        script = root / "bridge.R"
        script.write_text(_R_SCRIPT, encoding="utf-8")
        try:
            out = sp.run([exe, str(script), str(root)], capture_output=True, text=True, timeout=timeout, check=False)
        except sp.TimeoutExpired as e:
            raise RBridgeError(f"{command}: R took longer than {timeout:g} s (set MORIE_R_TIMEOUT)") from e
    if out.returncode != 0:
        raise RBridgeError(f"{command}: {out.stderr.strip() or 'R failed'}")
    return out.stdout


# ---------------------------------------------------------------------------
# Command-line words (the morie prompt) to Python values
# ---------------------------------------------------------------------------


def _word_value(word: str) -> Any:
    low = word.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(word)
    except ValueError:
        pass
    try:
        return float(word)
    except ValueError:
        pass
    if "~" in word:
        return word
    if "," in word:
        return [_word_value(p) for p in word.split(",") if p != ""]
    if word.lower().endswith((".csv", ".tsv", ".xlsx", ".xls")) and Path(word).is_file():
        from morie.fn import _frame_core as frame

        if word.lower().endswith((".xlsx", ".xls")):
            return frame.read_excel(word)
        return frame.read_csv(word, sep="\t" if word.lower().endswith(".tsv") else ",")
    return word


def parse_words(words: list[str]) -> tuple[list[Any], dict[str, Any]]:
    """Turn prompt words into arguments: name=value is a keyword, a .csv/.xlsx path is read as a
    data frame, 1,2,3 is a vector, numbers and true/false are converted, and text with ~ is a formula.
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for w in words:
        key, eq, val = w.partition("=")
        if eq and _R_NAME.match(key) and "~" not in key:
            kwargs[key] = _word_value(val)
        else:
            args.append(_word_value(w))
    return args, kwargs
