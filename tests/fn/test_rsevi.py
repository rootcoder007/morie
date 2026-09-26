"""rsevi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsevi import rsevi


def test_rsevi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsevi()
