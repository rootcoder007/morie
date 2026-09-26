"""rsshd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsshd import rsshd


def test_rsshd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsshd()
