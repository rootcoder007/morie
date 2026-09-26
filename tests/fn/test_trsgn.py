"""trsgn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trsgn import trsgn


def test_trsgn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trsgn()
