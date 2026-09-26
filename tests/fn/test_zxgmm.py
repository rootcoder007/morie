"""zxgmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxgmm import gmm_spatial


def test_zxgmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gmm_spatial(data=None)
