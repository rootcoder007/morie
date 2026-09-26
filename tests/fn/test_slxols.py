"""slxols is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slxols import slxols


def test_slxols_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slxols(y=None, X=None, W=None)
