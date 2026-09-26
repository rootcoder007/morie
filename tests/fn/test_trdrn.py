"""trdrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trdrn import trdrn


def test_trdrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trdrn()
