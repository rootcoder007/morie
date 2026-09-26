"""voroni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.voroni import voroni


def test_voroni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        voroni()
