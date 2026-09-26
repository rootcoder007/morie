"""trprd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trprd import trprd


def test_trprd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trprd()
