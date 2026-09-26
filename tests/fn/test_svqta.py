"""svqta is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svqta import quota_game


def test_svqta_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        quota_game(data=None)
