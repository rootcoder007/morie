"""trped is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trped import trped


def test_trped_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trped()
