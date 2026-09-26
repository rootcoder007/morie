"""sdm2slg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdm2slg import sdm2slg


def test_sdm2slg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdm2slg(y=None, X=None, W=None)
