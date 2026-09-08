"""Integration tests for the 30 Bayesian nonparametric inference modules."""

import importlib
import inspect

BNP_MODULES = [
    "bbstr", "bcntr", "bferg", "bnpht", "bnpqs", "brnst", "bspln", "bwavl",
    "crppr", "crpst", "dpgen", "dpkde", "dpmdn", "dpmix", "dpprr", "ewens",
    "gpclf", "gphyp", "gpkrn", "gprgr", "hdprc", "ibprc", "lddst", "neale",
    "polya", "polyt", "postc", "pyprr", "slcmx", "stbrk",
]


def _public_functions(mod):
    """Module-level functions defined IN the module, minus helpers."""
    return [
        fn
        for name, fn in vars(mod).items()
        if inspect.isfunction(fn)
        and fn.__module__ == mod.__name__
        and not name.startswith("_")
        and name != "cheatsheet"
    ]


def test_every_bnp_module_exports_a_callable():
    """Each module must define at least one public function with a
    docstring. The old version asserted callable(module), which is false
    for EVERY module -- it could never pass and tested nothing."""
    problems = []
    for name in BNP_MODULES:
        mod = importlib.import_module(f"morie.fn.{name}")
        fns = _public_functions(mod)
        if not fns:
            problems.append(f"{name}: no public function")
        elif not any(fn.__doc__ for fn in fns):
            problems.append(f"{name}: public function without docstring")
    assert not problems, problems


def test_bnp_modules_have_docstrings():
    for name in BNP_MODULES:
        mod = importlib.import_module(f"morie.fn.{name}")
        assert mod.__doc__, name


def test_the_ewens_sampler_produces_a_valid_partition():
    """Smoke-run one representative end to end: the sampled partition must
    cover {0..n-1} exactly once, and class_sizes must sum to n."""
    from morie.fn import _array_core as np

    from morie.fn.ewens import ewens_partition

    out = ewens_partition(20, theta=1.5, rng=np.random.default_rng(0))
    assert int(np.sum(out["class_sizes"])) == 20
    assert out["n_classes"] == len(out["partition"])
    flat = sorted(i for cls in out["partition"] for i in cls)
    assert flat == list(range(20))
