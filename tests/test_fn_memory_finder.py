"""morie.fn can serve its bundled sources from memory (MORIE_FN_NO_CACHE=1, or when no cache
directory is writable) instead of writing a 15 MB zip into the user cache on first import."""

import importlib
import inspect
import sys


def test_memory_finder_imports_and_exposes_source():
    import morie.fn as F

    F._install_memory_finder({"zzq_mem_probe": "def answer():\n    return 42\n"})
    try:
        mod = importlib.import_module("morie.fn.zzq_mem_probe")
        assert mod.answer() == 42
        assert "return 42" in inspect.getsource(mod.answer)
        assert mod.__spec__.origin == "morie-fnsrc:zzq_mem_probe.py"
    finally:
        sys.modules.pop("morie.fn.zzq_mem_probe", None)
