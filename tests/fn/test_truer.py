"""truer is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.truer import truer


def test_truer_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        truer()
