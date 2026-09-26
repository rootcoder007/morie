"""trspd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trspd import trspd


def test_trspd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trspd()
