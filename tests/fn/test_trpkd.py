"""trpkd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trpkd import trpkd


def test_trpkd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trpkd()
