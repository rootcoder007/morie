# SPDX-License-Identifier: AGPL-3.0-or-later
"""The input contract of the estimators: a frame, a CSV path, or a dict of columns (README)."""

from __future__ import annotations

import os
from typing import Any


def as_frame(obj: Any, *, name: str = "data"):
    """Return ``obj`` as a frame: a frame is returned as is, a path is read with
    :func:`morie.dataset.load_dataset`, a dict of columns becomes a frame.

    Examples
    --------
    >>> from morie._frames import as_frame
    >>> df = as_frame({"x": [1, 2, 3], "y": [0, 1, 0]})
    >>> (len(df), list(df.columns))
    (3, ['x', 'y'])
    """
    if isinstance(obj, str | os.PathLike):
        from morie.dataset import load_dataset

        return load_dataset(obj)
    if isinstance(obj, dict):
        from morie.fn import _frame_core as pd

        return pd.DataFrame(obj)
    if hasattr(obj, "columns") and hasattr(obj, "__len__"):
        return obj
    raise TypeError(f"{name} must be a data frame, a CSV path or a dict of columns, not {type(obj).__name__}")
