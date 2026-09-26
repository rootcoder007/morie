"""trcrw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trcrw import trcrw


def test_trcrw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trcrw()
