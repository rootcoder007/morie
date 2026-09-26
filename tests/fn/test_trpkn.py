"""trpkn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trpkn import trpkn


def test_trpkn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trpkn()
