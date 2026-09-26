"""mspro is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mspro import procrustes_orth


def test_mspro_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        procrustes_orth(X=None)
