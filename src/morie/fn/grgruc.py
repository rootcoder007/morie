# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRU cell forward pass (Géron's coupled-gate form)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gru_cell"]

_METHOD = "GRU cell forward pass"


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_gru_cell(x_t, h_prev, Wz, Wr, W):
    r"""One GRU step.

    .. math::
        z &= \sigma(W_z [h, x]) \\
        r &= \sigma(W_r [h, x]) \\
        \tilde h &= \tanh(W [r \odot h, x]) \\
        h_t &= (1-z) \odot h_{\text{prev}} + z \odot \tilde h

    A GRU is an LSTM with the input and forget gates *tied*: one gate
    ``z`` decides how much to keep and how much to overwrite, so there
    is no separate cell state and a third of the parameters disappear.

    Note the reset gate multiplies ``h`` *before* the matrix, i.e.
    ``W [r ⊙ h, x]``, which is what the source formula says.  The
    variant in :func:`morie.fn.grucl.gru_cell` applies ``r`` after the
    recurrent matrix (``r ⊙ (U h)``) and uses the opposite convention
    for ``z``; those are different functions for a non-diagonal ``U``,
    so this module implements the formula natively rather than
    delegating.

    Each matrix acts on a concatenation and so has shape ``(H, H + n)``.

    Parameters
    ----------
    x_t : array-like, shape (n,)
    h_prev : array-like, shape (H,)
    Wz, Wr, W : array-like, shape (H, H + n)

    Returns
    -------
    RichResult
        Payload keys ``h``, ``z``, ``r``, ``h_tilde``,
        ``update_fraction`` (mean of ``z``), ``estimate`` (= ``h``),
        ``n``, ``method``.

    References
    ----------
    Géron Ch 13, GRU section (Cho et al. 2014).

    Examples
    --------
    Zero weights put both gates at 0.5 and the candidate at 0, so the
    new state is exactly half the old one:

    >>> Z = [[0.0, 0.0]]
    >>> r = geron_gru_cell([0.0], [1.0], Z, Z, Z)
    >>> r["z"], r["r"], r["h_tilde"]
    ([0.5], [0.5], [0.0])
    >>> r["h"]
    [0.5]

    Force ``z`` to 0 (a large negative pre-activation on the update
    gate) and the state is carried through untouched -- the GRU's
    memory path:

    >>> r2 = geron_gru_cell([1.0], [0.7], [[0.0, -50.0]], Z, Z)
    >>> round(r2["h"][0], 10)
    0.7
    """
    x = np.asarray(x_t, dtype=float).ravel()
    h = np.asarray(h_prev, dtype=float).ravel()
    H, n = h.size, x.size
    if H == 0:
        raise ValueError("h_prev is empty; the cell needs at least one unit.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(h)):
        raise ValueError("x_t and h_prev must be finite.")

    mats = {}
    for name, M in (("Wz", Wz), ("Wr", Wr), ("W", W)):
        A = np.atleast_2d(np.asarray(M, dtype=float))
        if A.shape != (H, H + n):
            raise ValueError(
                f"{name} must have shape (H, H + n) = ({H}, {H + n}) to act on a "
                f"concatenated [h, x], got {A.shape}."
            )
        if not np.all(np.isfinite(A)):
            raise ValueError(f"{name} must be finite.")
        mats[name] = A

    hx = np.concatenate([h, x])
    z = _sigmoid(mats["Wz"] @ hx)
    r = _sigmoid(mats["Wr"] @ hx)
    h_tilde = np.tanh(mats["W"] @ np.concatenate([r * h, x]))
    h_new = (1.0 - z) * h + z * h_tilde

    return RichResult(
        title="GRU cell",
        summary_lines=[("Units", int(H)), ("Update gate mean", float(z.mean()))],
        payload={
            "h": h_new.tolist(),
            "z": z.tolist(),
            "r": r.tolist(),
            "h_tilde": h_tilde.tolist(),
            "update_fraction": float(z.mean()),
            "estimate": h_new.tolist(),
            "n": int(H),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgruc: h = (1-z)*h_prev + z*tanh(W[r*h, x]); one tied gate instead of the LSTM's two"


# compact alias per ledger/NAMING.md
gerongrucell = geron_gru_cell
