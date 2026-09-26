"""svjhn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svjhn import johnston_power


def test_svjhn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        johnston_power(data=None)
