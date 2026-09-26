"""msprb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msprb import procrustes_obl


def test_msprb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        procrustes_obl(X=None)
