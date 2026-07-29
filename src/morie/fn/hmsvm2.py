# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Save and load PyTorch model state_dict."""

import os

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_save_load_pytorch"]


def geron_save_load_pytorch(model, path, verify=True):
    """
    Save and load a model state dict.

    Formula: torch.save(model.state_dict(), path); model.load_state_dict(torch.load(path))

    A numpy-native stand-in for the torch API (torch is not a dependency
    of morie), implementing what the idiom actually guarantees: the
    *parameters* are serialised, not the class, and reloading them must
    reproduce every tensor exactly.

    So the round trip is performed and checked entry by entry. Two things
    are enforced that the torch idiom is famous for getting wrong:

    * **keys must match exactly on reload** -- a missing or unexpected key
      is an error, never a silently half-initialised model;
    * **dtypes and shapes are preserved**, and equality is checked
      bit-exactly rather than with a tolerance, because saving and loading
      is not supposed to be an approximation.

    `path` must be given explicitly; nothing is ever written to a default
    or user-home location.

    Parameters
    ----------
    model : mapping or sequence of arrays
        The state dict (name -> tensor), or a list of tensors which are
        named ``param_0, param_1, ...``.
    path : str
        Destination file (``.npz``). Its directory must already exist.
    verify : bool, default True
        Reload and compare after writing.

    Returns
    -------
    result : RichResult
        Keys: path, keys, shapes, n_params, bytes, loaded, exact,
        max_diff, estimate, n, method.

    Examples
    --------
    >>> import os, tempfile, numpy as np
    >>> d = tempfile.mkdtemp()
    >>> p = os.path.join(d, "state.npz")
    >>> sd = {"w1": np.array([[1.0, 2.0], [3.0, 4.0]]), "b1": np.array([0.5, -0.5])}
    >>> r = geron_save_load_pytorch(sd, p)
    >>> bool(r["exact"])
    True
    >>> int(r["n_params"])
    6
    >>> sorted(r["keys"])
    ['b1', 'w1']
    >>> bool(np.array_equal(r["loaded"]["w1"], sd["w1"]))
    True
    >>> round(float(r["max_diff"]), 12)
    0.0
    >>> os.path.exists(p)
    True

    References
    ----------
    Géron Ch 10
    """
    if isinstance(model, dict):
        state = {str(k): np.asarray(v) for k, v in model.items()}
    else:
        try:
            items = list(model)
        except TypeError:
            raise ValueError("geron_save_load_pytorch: model must be a state-dict mapping or a sequence of tensors") from None
        state = {f"param_{i}": np.asarray(v) for i, v in enumerate(items)}
    if not state:
        raise ValueError("geron_save_load_pytorch: the state dict is empty; there is nothing to save")
    for k, v in state.items():
        if v.size == 0:
            raise ValueError(f"geron_save_load_pytorch: entry {k!r} is empty")
        if v.dtype.kind not in "fiub":
            raise ValueError(f"geron_save_load_pytorch: entry {k!r} has non-numeric dtype {v.dtype}")

    p = str(path)
    if not p:
        raise ValueError("geron_save_load_pytorch: path is required; nothing is written to a default location")
    directory = os.path.dirname(os.path.abspath(p))
    if not os.path.isdir(directory):
        raise ValueError(f"geron_save_load_pytorch: directory {directory!r} does not exist")
    if not p.endswith(".npz"):
        p = p + ".npz"

    np.savez(p, **state)

    loaded = {}
    exact = None
    max_diff = None
    if verify:
        with np.load(p) as z:
            loaded = {k: z[k] for k in z.files}
        missing = sorted(set(state) - set(loaded))
        unexpected = sorted(set(loaded) - set(state))
        if missing or unexpected:
            raise ValueError(
                f"geron_save_load_pytorch: reloaded state dict does not match -- missing {missing}, unexpected {unexpected}"
            )
        exact = True
        max_diff = 0.0
        for k, v in state.items():
            w = loaded[k]
            if w.shape != v.shape:
                raise ValueError(f"geron_save_load_pytorch: entry {k!r} reloaded with shape {w.shape}, expected {v.shape}")
            if w.dtype != v.dtype:
                raise ValueError(f"geron_save_load_pytorch: entry {k!r} reloaded as {w.dtype}, expected {v.dtype}")
            exact = exact and bool(np.array_equal(w, v))
            max_diff = max(max_diff, float(np.max(np.abs(w.astype(float) - v.astype(float)))))

    n_params = int(sum(v.size for v in state.values()))
    nbytes = int(sum(v.nbytes for v in state.values()))

    return RichResult(
        title="State-dict save / load round trip",
        summary_lines=[
            ("Path", p),
            ("Entries", len(state)),
            ("Parameters", n_params),
            ("Exact round trip", exact if exact is not None else "not verified"),
        ],
        interpretation=(
            "Saving the state dict rather than the object keeps the checkpoint portable: it is just "
            "named tensors, and the code that rebuilds the architecture stays in your source."
        ),
        payload={
            "path": p,
            "keys": sorted(state),
            "shapes": {k: tuple(v.shape) for k, v in state.items()},
            "dtypes": {k: str(v.dtype) for k, v in state.items()},
            "n_params": n_params,
            "bytes": nbytes,
            "loaded": loaded,
            "exact": exact,
            "max_diff": max_diff,
            "estimate": float(n_params),
            "n": int(len(state)),
            "method": "state_dict serialised to .npz and reloaded, with key/shape/dtype and bit-exact value checks",
        },
    )


def cheatsheet():
    return "hmsvm2: Save and load PyTorch model state_dict"
