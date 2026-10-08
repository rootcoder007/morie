"""morie.fn -- lazy-loaded function modules (PEP 562).

The package hosts ~13.5k callable modules exposing ~18.5k names.  ``__init__`` reads the
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

import importlib as _importlib
import importlib.abc
import importlib.util
import json as _json
import os as _os
import sys as _sys
import types as _types

# the stdlib imports are private: `from morie.fn import os` must be an ImportError, not the stdlib module
_FN_DIR = _os.path.dirname(__file__)
_MAP_PATH = _os.path.join(_FN_DIR, "_lazy_map.json")
with open(_MAP_PATH) as _f:
    _LAZY_MAP = _json.load(_f)

# module -> every public name it exports through the map
_MODULE_NAMES: "dict[str, list[str]]" = {}
for _n, _m in _LAZY_MAP.items():
    _MODULE_NAMES.setdefault(_m, []).append(_n)


def _bind_module(modname, mod):
    """Bind every name the map assigns to ``mod``.

    Importing ``morie.fn.<modname>`` sets ``morie.fn.<modname>`` to the
    module object. When a module exports a callable of its own name (the
    convention here), binding only the requested name left the sibling
    shadowed by the module for the rest of the process, and the order
    of first access decided which. Binding all of them at once closes
    that.
    """
    g = globals()
    for n in _MODULE_NAMES.get(modname, ()):
        obj = getattr(mod, n, None)
        if obj is not None and not isinstance(obj, _types.ModuleType):
            g[n] = obj


def __getattr__(name):
    if name in _LAZY_MAP:
        mod = _importlib.import_module("." + _LAZY_MAP[name], package=__name__)
        _bind_module(_LAZY_MAP[name], mod)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _FnPackage(_types.ModuleType):
    """The package's module type: an explicit ``import morie.fn.X`` binds
    the submodule as an attribute, which would shadow the callable ``X``
    the map exports from it; keep the callable instead."""

    def __setattr__(self, name, value):
        if (
            isinstance(value, _types.ModuleType)
            and name in _LAZY_MAP
            and getattr(value, "__name__", "") == __name__ + "." + _LAZY_MAP[name]
        ):
            obj = getattr(value, name, None)
            if obj is not None and not isinstance(obj, _types.ModuleType):
                _bind_module(_LAZY_MAP[name], value)
                return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _FnPackage


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
    xdg = _os.environ.get("XDG_CACHE_HOME")
    if xdg:
        dirs.append(_os.path.join(xdg, "morie"))
    home = _os.path.expanduser("~")
    if home and home not in ("", "~"):
        dirs.append(_os.path.join(home, ".cache", "morie"))
    uid = _os.getuid() if hasattr(_os, "getuid") else "u"
    dirs.append(_os.path.join(tempfile.gettempdir(), f"morie-cache-{uid}"))
    return dirs


def _owned_private(path, want_dir):
    """True when ``path`` is ours alone: not a symlink, owned by this uid, not group/world-writable.

    The shared ``/tmp/morie-cache-<uid>`` can be pre-created by another user with a planted
    ``fnsrc-<tag>.zip`` (the tag is computable from the public wheel), so a cache directory or
    zip is only trusted when it passes this check. Same-user tampering is out of scope: that
    user can already edit the installed package. Windows has no uid; there the per-user
    profile directories are private by default.
    """
    import stat

    try:
        st = _os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if want_dir != stat.S_ISDIR(st.st_mode) or (not want_dir and not stat.S_ISREG(st.st_mode)):
        return False
    if not hasattr(_os, "getuid"):
        return True
    return st.st_uid == _os.getuid() and not st.st_mode & 0o022


def _decompress_fnsrc(xz_path):
    import lzma

    with lzma.open(xz_path, "rt", encoding="utf-8") as fh:
        return _json.load(fh)  # {short: source}


def _write_cache_zip(target, sources):
    import tempfile
    import zipfile

    _os.makedirs(_os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_os.path.dirname(target), suffix=".tmp")
    _os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for short, src in sources.items():
                zf.writestr(short + ".py", src)
        _os.replace(tmp, target)  # atomic publish -- other processes never see a partial zip
    finally:
        try:
            if _os.path.exists(tmp):
                _os.remove(tmp)
        except OSError:
            pass


def _install_fnsrc():
    """Make morie.fn.<short> importable from whichever archive layout shipped."""
    # Layout 2: a ready-made zip next to us -> just put it on the path.
    zip_path = _os.path.join(_FN_DIR, "_fnsrc.zip")
    if _os.path.isfile(zip_path):
        if zip_path not in __path__:
            __path__.append(zip_path)
        return

    # Layout 3: solid-lzma archive -> decompress once into an on-disk cache zip.
    xz_path = _os.path.join(_FN_DIR, "_fnsrc.json.xz")
    if not _os.path.isfile(xz_path):
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
        cz = _os.path.join(base, cache_name)
        if _os.path.isfile(cz) and _owned_private(base, True) and _owned_private(cz, False):
            if cz not in __path__:
                __path__.append(cz)
            return

    # Build it once (atomic, so concurrent interpreters are safe).
    try:
        sources = _decompress_fnsrc(xz_path)
    except Exception:
        return  # corrupt/unreadable archive -> leave loose-file behaviour
    # MORIE_FN_NO_CACHE=1: never write the 15 MB zip; serve the sources from memory
    # (inspect.getsource() still works through the loader).
    if _os.environ.get("MORIE_FN_NO_CACHE", "").strip().lower() in ("1", "true", "yes"):
        _install_memory_finder(sources)
        return
    for base in _candidate_cache_dirs():
        cz = _os.path.join(base, cache_name)
        try:
            _os.makedirs(base, mode=0o700, exist_ok=True)
            if not _owned_private(base, True):
                continue  # someone else's (or a world-writable) directory: never write or load there
            _write_cache_zip(cz, sources)
        except OSError:
            continue
        if cz not in __path__:
            __path__.append(cz)
        return

    # No writable cache dir anywhere: serve the sources from memory rather than writing a
    # process-private zip that nothing cleans up.
    _install_memory_finder(sources)


class _MemLoader(importlib.abc.SourceLoader):
    """SourceLoader over the in-memory source dict: the import system compiles and runs the
    module itself (no eval/exec call in this package); inspect.getsource() / describe() read
    get_data()."""

    def __init__(self, short, sources):
        self.short = short
        self._sources = sources

    def get_filename(self, fullname):
        return f"morie-fnsrc:{self.short}.py"

    def get_data(self, path):
        return self._sources[self.short].encode("utf-8")

    def path_stats(self, path):
        raise OSError("no bytecode cache for in-memory sources")

    def is_package(self, fullname):
        return False


class _Finder(importlib.abc.MetaPathFinder):
    """Meta-path finder for ``morie.fn.<short>`` served from memory. One per process: later
    calls to ``_install_memory_finder`` merge their sources into the existing instance."""

    def __init__(self, sources):
        self.sources = dict(sources)

    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + "."
        if not fullname.startswith(prefix):
            return None
        short = fullname[len(prefix) :]
        if short not in self.sources:
            return None
        return importlib.util.spec_from_loader(
            fullname, _MemLoader(short, self.sources), origin=f"morie-fnsrc:{short}.py"
        )


def _install_memory_finder(sources):
    """Import ``morie.fn.<short>`` from the decompressed source dict, no file written."""
    import sys as _sys

    for f in _sys.meta_path:
        if isinstance(f, _Finder):
            f.sources.update(sources)
            return f
    finder = _Finder(sources)
    _sys.meta_path.append(finder)
    return finder


_install_fnsrc()


def load_all(ignore_errors: bool = True):
    """Eagerly import every ``morie.fn`` callable.

    Opt-in only -- the package default is lazy (``import morie.fn`` stays
    ~0.05 s / ~28 MB). Materialising all ~36k callables costs roughly 20 s and
    ~400 MB, so call this only for exploration or for static-analysis / graph
    tooling that needs every module resolved.

    Returns ``(ok, failed)``: counts of callables imported and of those that
    failed (typically because an optional third-party dependency is absent).
    Set ``ignore_errors=False`` to re-raise the first failure instead.
    """
    ok = 0
    failed = 0
    for _name in _LAZY_MAP:
        try:
            __getattr__(_name)
            ok += 1
        except Exception:
            if not ignore_errors:
                raise
            failed += 1
    return ok, failed
