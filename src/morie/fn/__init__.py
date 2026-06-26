"""morie.fn -- lazy-loaded function modules (PEP 562).

The package hosts ~36k auto-generated callable modules.  ``__init__`` reads the
symbol->submodule map from sibling ``_lazy_map.json`` and resolves callables on
first access via :pep:`562` ``__getattr__``.

The per-callable implementation modules ship in one of three layouts; this
module makes ``morie.fn.<short>`` importable from whichever is present, with no
change to the lazy ``__getattr__`` below:

  1. loose ``<short>.py`` files       -- dev tree / sdist.  The package
     directory is already on ``__path__``, so nothing to do.
  2. ``_fnsrc.zip`` (deflate)         -- appended to ``__path__`` for zipimport.
  3. ``_fnsrc.json.xz`` (solid lzma, ~3 MB vs ~28 MB) -- the small wheel.  It is
     decompressed *once* into an on-disk cache ``.zip`` which is put on
     ``__path__``; subsequent interpreters reuse the cache (low steady-state
     RAM).  If no cache directory is writable, an in-memory finder compiles the
     modules straight from the decompressed sources.

Cold ``import morie.fn`` stays ~1 s; the layout-3 cache is built lazily and only
once per (version of the) archive.
"""

import importlib
import importlib.util
import json
import os
import sys

_FN_DIR = os.path.dirname(__file__)
_MAP_PATH = os.path.join(_FN_DIR, "_lazy_map.json")
with open(_MAP_PATH) as _f:
    _LAZY_MAP = json.load(_f)


def __getattr__(name):
    if name in _LAZY_MAP:
        mod = importlib.import_module("." + _LAZY_MAP[name], package=__name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_MAP) + ["__getattr__", "__dir__"])


__all__ = sorted(_LAZY_MAP)


# --------------------------------------------------------------------------- #
# Per-callable source resolution (layouts 2 and 3).                           #
# --------------------------------------------------------------------------- #

# Kept populated only when we fall back to the in-memory finder (no writable
# cache dir); otherwise the decompressed sources are freed after the cache zip
# is written. describe.py reaches the source via the import system (get_source).
_inmem_sources: "dict[str, str] | None" = None


def _candidate_cache_dirs():
    import tempfile

    dirs = []
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        dirs.append(os.path.join(xdg, "morie"))
    home = os.path.expanduser("~")
    if home and home not in ("", "~"):
        dirs.append(os.path.join(home, ".cache", "morie"))
    uid = os.getuid() if hasattr(os, "getuid") else "u"
    dirs.append(os.path.join(tempfile.gettempdir(), f"morie-cache-{uid}"))
    return dirs


def _decompress_fnsrc(xz_path):
    import lzma

    with lzma.open(xz_path, "rt", encoding="utf-8") as fh:
        return json.load(fh)  # {short: source}


def _write_cache_zip(target, sources):
    import tempfile
    import zipfile

    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for short, src in sources.items():
                zf.writestr(short + ".py", src)
        os.replace(tmp, target)  # atomic publish -- other processes never see a partial zip
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


class _InMemoryFnLoader:
    """Loader compiling a single morie.fn.<short> from an in-RAM source string."""

    def __init__(self, fullname, source):
        self._fullname = fullname
        self._source = source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self._source, f"<{self._fullname}>", "exec"), module.__dict__)

    def get_source(self, fullname):
        return self._source


class _InMemoryFnFinder:
    """MetaPathFinder serving morie.fn.<short> from a decompressed source map.
    Used only when no cache directory is writable."""

    def __init__(self, sources):
        self._sources = sources
        self._prefix = __name__ + "."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix):
            return None
        src = self._sources.get(fullname[len(self._prefix):])
        if src is None:
            return None
        return importlib.util.spec_from_loader(fullname, _InMemoryFnLoader(fullname, src))


def _install_fnsrc():
    """Make morie.fn.<short> importable from whichever archive layout shipped."""
    global _inmem_sources

    # Layout 2: a ready-made zip next to us -> just put it on the path.
    zip_path = os.path.join(_FN_DIR, "_fnsrc.zip")
    if os.path.isfile(zip_path):
        if zip_path not in __path__:
            __path__.append(zip_path)
        return

    # Layout 3: solid-lzma archive -> decompress once into an on-disk cache zip.
    xz_path = os.path.join(_FN_DIR, "_fnsrc.json.xz")
    if not os.path.isfile(xz_path):
        return  # Layout 1 (loose files) -- the package dir is already on __path__.

    import hashlib

    try:
        with open(xz_path, "rb") as fh:
            tag = hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        tag = "x"
    cache_name = f"fnsrc-{tag}.zip"

    # Fast path: a cache zip from a previous run already exists.
    for base in _candidate_cache_dirs():
        cz = os.path.join(base, cache_name)
        if os.path.isfile(cz):
            if cz not in __path__:
                __path__.append(cz)
            return

    # Build it once (atomic, so concurrent interpreters are safe).
    try:
        sources = _decompress_fnsrc(xz_path)
    except Exception:
        return  # corrupt/unreadable archive -> leave loose-file behaviour
    for base in _candidate_cache_dirs():
        cz = os.path.join(base, cache_name)
        try:
            _write_cache_zip(cz, sources)
        except OSError:
            continue
        if cz not in __path__:
            __path__.append(cz)
        return

    # No writable cache dir anywhere -> compile from memory.
    _inmem_sources = sources
    sys.meta_path.insert(0, _InMemoryFnFinder(sources))


_install_fnsrc()
