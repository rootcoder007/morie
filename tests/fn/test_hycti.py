"""hycti is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hycti import hycti


def test_hycti_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hycti()
