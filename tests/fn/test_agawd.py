"""agawd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agawd import agawd


def test_agawd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agawd()
