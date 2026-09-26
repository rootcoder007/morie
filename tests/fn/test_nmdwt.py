"""nmdwt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmdwt import dwnominate_trend


def test_nmdwt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dwnominate_trend(data=None)
