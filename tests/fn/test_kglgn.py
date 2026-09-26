"""kglgn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kglgn import lognormal_kriging


def test_kglgn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lognormal_kriging(values=None, x=None)
