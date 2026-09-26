"""csenv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csenv import csenv


def test_csenv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csenv()
