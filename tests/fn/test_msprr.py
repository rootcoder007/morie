"""msprr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msprr import procrustes_resid


def test_msprr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        procrustes_resid(X=None)
