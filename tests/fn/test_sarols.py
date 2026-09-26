"""sarols is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sarols import sarols


def test_sarols_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sarols(y=None, X=None, W=None)
