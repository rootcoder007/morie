"""aflai is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aflai import aflai


def test_aflai_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aflai()
