"""swrank is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swrank import swrank


def test_swrank_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swrank(W=None)
