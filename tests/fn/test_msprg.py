"""msprg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msprg import procrustes_gen


def test_msprg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        procrustes_gen(X=None)
