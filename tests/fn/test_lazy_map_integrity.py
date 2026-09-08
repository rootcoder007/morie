"""The lazy map must not point a public symbol at a shadowing module.

morie.fn resolves every public callable through ``_lazy_map.json``. Where
two modules define a function of the same name, the map picks one, and
the other becomes reachable only by direct module import.

Six symbols -- adamw, admm, bart, ets, lbfgs and tfidf -- were pointing at
an abbreviated module holding a generated mean/se placeholder, so
``morie.fn.ets`` returned the mean of the series instead of running
Holt-Winters. Only ``bart`` had a test sharp enough to notice.

This guard states the rule that fixes them: if a module is named after
the symbol *and* defines it, that module wins. It says nothing about
symbols whose same-named module defines something else (``morie.fn.hdi``
correctly comes from hdint, because hdi.py defines
``highest_density_interval``), and nothing about dffits, where the short
name is the placeholder and the map already points at the real dffts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

FN_DIR = Path(__file__).resolve().parents[2] / "src" / "morie" / "fn"

# Symbols where the map deliberately prefers the differently named module.
# Both entries were checked by hand; neither target is a placeholder.
#
#   dffits  dffits.py holds the generated placeholder and dffts.py the real
#           influence measure, so the map is right to prefer dffts.
#   vecm    vecm.py handles two series and returns a plain dict; vecmf.py
#           takes a (T, k) panel with a cointegration rank and returns a
#           RichResult. Both are real; the map prefers the general one.
KNOWN_INVERTED = {"dffits", "vecm"}


def _defined_functions(module: str) -> set[str]:
    tree = ast.parse((FN_DIR / f"{module}.py").read_text())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_symbol_resolves_to_its_own_module_when_that_module_defines_it():
    lazy = json.loads((FN_DIR / "_lazy_map.json").read_text())
    offenders = []
    for symbol, module in lazy.items():
        if module == symbol or symbol in KNOWN_INVERTED:
            continue
        own = FN_DIR / f"{symbol}.py"
        if own.exists() and symbol in _defined_functions(symbol):
            offenders.append(f"{symbol} -> {module}, but {symbol}.py defines {symbol}")
    assert not offenders, "lazy map shadows a same-named implementation:\n" + "\n".join(offenders)


def test_every_target_module_exists_and_defines_its_symbol():
    lazy = json.loads((FN_DIR / "_lazy_map.json").read_text())
    missing = [f"{s} -> {m}" for s, m in lazy.items() if not (FN_DIR / f"{m}.py").exists()]
    assert not missing, "lazy map points at modules that do not exist:\n" + "\n".join(missing)
