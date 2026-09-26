"""hullar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hullar import hullar


def test_hullar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hullar()
