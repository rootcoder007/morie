"""msprp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msprp import procrustes_part


def test_msprp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        procrustes_part(X=None)
