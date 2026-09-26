"""msprc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msprc import procrustes_corr


def test_msprc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        procrustes_corr(X=None)
